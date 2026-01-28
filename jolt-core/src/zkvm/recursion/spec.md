# Jolt Recursion via SNARK Composition

## 1. Motivation & Overview

### 1.1 Why Recursion

Recursion enables two critical capabilities:

1. **Proof Aggregation**: Maintain a single succinct proof verifying entire blockchain state, enabling light clients and efficient synchronization.

2. **On-Chain Verification**: Produce proofs with single-digit millisecond verification on mobile and reasonable EVM gas costs.

### 1.2 The Approach

The Jolt verifier compiled to RISC-V requires ~1.5 billion cycles. Our target is <10 million cycles (~150× reduction).

We decompose the verifier:

$$\mathcal{V}_{\text{Jolt}} = \mathcal{V}_{\text{light}} \circ \mathcal{V}_{\text{Dory}}$$

where $\mathcal{V}_{\text{Dory}}$ is the verification logic for the Dory polynomial commitment scheme. This logic is dominated by expensive group operations (scalar multiplication, exponentiation, pairing).

Instead of the Jolt verifier executing these operations directly (which is expensive in the R1CS/AIR constraint model), we offload them to a **bespoke recursion SNARK**.

The recursion SNARK proves that the execution trace of the Dory verifier is correct.

The expensive operations (from Dory PCS verification):

| Operation | Description |
|-----------|-------------|
| G1 Scalar Mul | $[k]P$ for $P \in \mathbb{G}_1$ |
| G2 Scalar Mul | $[k]Q$ for $Q \in \mathbb{G}_2$ |
| GT Exponentiation | $a^k$ for $a \in \mathbb{G}_T$ |
| GT Multiplication | $a \cdot b$ for $a, b \in \mathbb{G}_T$ |
| Multi-Pairing | $\prod_i e(P_i, Q_i)$ |

### 1.3 Architecture: Dory-Proof-Ground-Truth, AST-Driven Wiring, External Pairing Boundary

We want the **Dory proof itself** (plus the public verifier setup and transcript-derived scalars) to be the ground truth.
The recursion SNARK proves the correctness of *all non-pairing group computation* performed by Dory verification, and we leave
the **final pairing check** to the outside verifier.

This removes the need for prover-provided intermediate “hints” of the form “here is some internal group element”, while still
avoiding an in-circuit pairing implementation.

Concretely:

1. **Witness**: The recursion prover commits to a witness encoding the complete non-pairing execution trace of Dory verification
   (G1/G2/GT scalar-muls, adds, GT mul/exp, plus internal packed traces where applicable).
2. **Operation constraints (Stage 1 + Stage 2)**: For every traced operation instance, we prove “this op is computed correctly in isolation”
   via type-specific sumchecks.
   - Stage 1 proves the packed GT exponentiation constraints (`SumcheckId::GtExp`).
   - Stage 2 proves all remaining constraint families (GT mul, G1/G2 scalar mul, G1/G2 add, MultiMillerLoop, ...), batched into one sumcheck proof.
3. **Internal trace consistency (Stage 2)**: Some witnesses contain “shifted” columns (e.g. `rho_next`, `A_next`) that are redundant.
   We enforce the one-step “next” relation via dedicated shift sumchecks:
   - `SumcheckId::GtShift` for packed GT exponentiation (\(\rho_{\text{next}}(s,x)=\rho(s+1,x)\))
   - `SumcheckId::ShiftG1ScalarMul` / `SumcheckId::ShiftG2ScalarMul` for scalar-mul traces (\(A_{\text{next}}(i)=A(i+1)\))
   - `SumcheckId::ShiftMultiMillerLoop` for Multi-Miller loop packed traces (\(f_{\text{next}}(s,x)=f(s+1,x)\), \(T_{\text{next}}(s)=T(s+1)\))
4. **Wiring / copy constraints (Stage 2)**: We prove that the output of each operation is exactly the input consumed by downstream
   operations, so the witness represents a single coherent computation DAG (not a bag of unrelated correct ops).
   > **Implementation status**: AST-derived wiring/boundary constraints are **implemented** and enabled by default. They run as additional
   > sumcheck instances appended at the end of the Stage 2 batched sumcheck (see `jolt-core/src/zkvm/recursion/{gt,g1,g2}/wiring.rs` and
   > `jolt-core/src/zkvm/recursion/verifier.rs`).
5. **Boundary outputs for the outside verifier**: The recursion SNARK exposes the **three (G1,G2) pairing input pairs** used by Dory’s
   final optimized check (a 3-way multi-pairing), and the corresponding GT “rhs” value (serialized as `PairingBoundary::rhs`). The outside
   verifier then computes the multi-pairing and checks equality itself.

#### Why AST (and not prover-supplied wiring metadata)

Dory already records the full verification computation as an AST/DAG where:
- nodes are typed group operations producing `ValueId`s, and
- edges are the `ValueId` dependencies (inputs) of each node.

We use this AST as the **authoritative topology source** for wiring constraints: every operation’s input `ValueId`s are wired to the
corresponding producer outputs. This prevents the prover from “choosing” an easier wiring graph.

#### Example: GT Homomorphic Combination

Suppose the proof contains GT elements $A, B, C$ and the verifier needs to check:
$$D = m \cdot A + n \cdot B + p \cdot C$$
(using additive notation for clarity, though GT is multiplicative).

The witness trace includes all intermediate values: $m \cdot A$, $n \cdot B$, $p \cdot C$, $m \cdot A + n \cdot B$, and finally $D$.

We do **not** expose these intermediate values to the verifier. Instead, we enforce the chain of computation via wiring constraints:

1.  **GT Exp 1**: Public input $A$, scalar $m$. Output $O_1$ (witness).
2.  **GT Exp 2**: Public input $B$, scalar $n$. Output $O_2$ (witness).
3.  **GT Exp 3**: Public input $C$, scalar $p$. Output $O_3$ (witness).
4.  **GT Mul 1**: Inputs $I_{4a}, I_{4b}$. Output $O_4$.
    - Wiring: $I_{4a} = O_1$ (copy constraint)
    - Wiring: $I_{4b} = O_2$ (copy constraint)
5.  **GT Mul 2**: Inputs $I_{5a}, I_{5b}$. Output $O_5$.
    - Wiring: $I_{5a} = O_4$ (copy constraint)
    - Wiring: $I_{5b} = O_3$ (copy constraint)
6.  **Final Check**: $O_5 = D$ (boundary constraint).

The verifier only knows $A, B, C, D, m, n, p$ and the circuit topology. The sumchecks prove that valid witnesses exist satisfying all operation and wiring constraints.

### 1.4 Protocol Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: Packed GT Exp Sumcheck                                    │
│  ─────────────────────────────                                      │
│  Prove packed GT exponentiation constraints                           │
│  Output: openings at `r*` (rho, rho_next, quotient); shift checked in Stage 2 │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2: Batched Constraint Sumchecks                               │
│  ────────────────────────────────────                               │
│  Includes:                                                           │
│  - Shift sumchecks for redundant “next” columns                       │
│    (`GtShift`, `ShiftG1ScalarMul`, `ShiftG2ScalarMul`, `ShiftMultiMillerLoop`) │
│  - Packed GT exp claim reduction to a shared `r_x` (`GtExpClaimReduction`)     │
│  - All other op constraints (GT mul, G1/G2 scalar mul, G1/G2 add, MultiMillerLoop, ...)│
│  - Wiring/boundary constraints (implemented; appended last)           │
│  Output: all virtual openings at a shared point `r_x`                │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3: Prefix Packing Reduction                                  │
│  ───────────────────────────────────                                │
│  Goal: connect Stage-2 virtual openings of many native-size witness  │
│  polynomials to ONE PCS opening of ONE packed multilinear.           │
│                                                                      │
│  - Derive a public, canonical prefix-packing layout                  │
│  - Sample fresh packing challenges `r_pack` (Fiat–Shamir)            │
│  - Compute `packed_eval` from Stage-2 virtual claims                 │
│  Output: `stage3_packed_eval` and packed opening point `r_full`      │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PCS Opening Proof (Hyrax over Grumpkin)                             │
│  ───────────────────────────────────                                │
│  Goal: prove the committed packed polynomial opens correctly at      │
│  `r_full` (and any other accumulated openings).                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4.1 End-to-end fused modes (GT / G1 / G2)

The recursion system supports **end-to-end fused** execution paths for GT, G1, and G2. These modes change:

- the **Stage 2 instance set** (which sumcheck instances run),
- the **Stage 2 challenge layout** (some instances use extra “constraint-index” variables `c`), and
- the **Stage 3 prefix-packing layout** (fused virtual openings are consumed instead of per-instance ones).

These modes are controlled by boolean flags carried in `RecursionVerifierInput` (not just environment variables), because the guest verifier
cannot safely rely on host environment variables.

#### High-level intent

- **Legacy (non-fused)**: each operation instance contributes **per-instance** virtual openings at the Stage-2 point `r_x` (or a suffix thereof).
- **Fused**: a whole family contributes **one set of fused openings**, indexed by an additional constraint-index variable segment `c`.
  Fused wiring/shift instances **reuse** those cached openings and therefore do **not** add new PCS openings.

#### Stage 2 ordering constraints (soundness-critical)

Stage 2 is a single `BatchedSumcheck`, but the *order of instances inside it matters* because later instances may read cached openings
produced by earlier ones.

- **GT fused end-to-end** (`enable_gt_fused_end_to_end`):
  - `FusedGtExpStage2Openings*` must run **before** `FusedGtShift*` and any fused GT wiring that consumes `gt_exp_rho_fused()` /
    `gt_exp_quotient_fused()` at the Stage-2 point.
  - This is enforced by instance ordering in `RecursionProver::prove_stage2` / `RecursionVerifier::verify_stage2`.

- **G1 fused scalar-mul / shift**:
  - `FusedG1ScalarMul*` must run **before** `FusedShiftG1ScalarMul*`, because the shift check intentionally caches **no new openings**
    and instead reuses the scalar-mul cached openings under `SumcheckId::G1ScalarMul`.

- **G2 fused scalar-mul / shift**:
  - `FusedG2ScalarMul*` must run **before** `FusedShiftG2ScalarMul*` for the same reason (`SumcheckId::G2ScalarMul`).

- **Fused wiring backends** (`gt/fused_wiring.rs`, `g1/fused_wiring.rs`, `g2/fused_wiring.rs`) are appended **last** in Stage 2.
  They are “openings consumers”: they read cached openings but do not create new ones.

#### Split-\(k\) / dummy-bit convention (shared across fused families)

Fused wiring uses a **common** constraint-index arity \(k_{\text{common}}\) (e.g. \(k_{GT}\), \(k_{G1}\), \(k_{G2}\)) that is the max
over the relevant families (e.g. scalar-mul vs add). Each family’s committed fused polynomials remain **family-local** (smaller \(k\)),
and are embedded into the common domain by replicating across **dummy low bits**:

- **dummy bits are low bits** of `c_common`
- **family bits are the suffix** of `c_common`
- selectors use \(\beta(\text{dummy})\cdot \mathrm{Eq}(c_{\text{tail}}, \text{idx})\), where \(\beta(\text{dummy}) = 2^{-\text{dummy}}\)

This convention is implemented for:

- GT: `gt/fused_stage2_openings.rs`, `gt/fused_wiring.rs`
- G1: `g1/fused_scalar_multiplication.rs`, `g1/fused_wiring.rs`, `g1/indexing.rs`
- G2: `g2/fused_scalar_multiplication.rs`, `g2/fused_wiring.rs`, `g2/indexing.rs`

#### Opening-point normalization for fused scalar-mul

In fused G1/G2 scalar-mul, the sumcheck runs over `(step, c_common)` (to align with wiring), but the committed fused witness tables are
family-local over `(step, c_smul)`. Therefore the opening point used for caching openings under `SumcheckId::{G1ScalarMul,G2ScalarMul}`
is formed by **dropping the dummy low c-bits** and keeping only the `c` tail:

- opening point = `(step_vars, c_tail)` (in big-endian point order for the PCS accumulator)

This is why the fused shift instances can safely reuse openings: they see the same Stage-2 challenges and normalize to the same opening point.

#### External step (outside this SNARK): final Dory multi-pairing check

After the recursion SNARK verifies, the outside verifier performs Dory’s final optimized check:
a **single 3-way multi-pairing** equality. The recursion SNARK provides the required three input pairs
\((p1\_g1,p1\_g2),(p2\_g1,p2\_g2),(p3\_g1,p3\_g2)\) as public outputs, and the outside verifier computes the
pairing product and compares it against the (public) GT rhs.

#### GT staging note (why GTExp is its own Stage 1 today)

Packed GT exponentiation has a “next-state” table \(\rho_{\text{next}}(s,x)\) that must satisfy the **shift** relation
\(\rho_{\text{next}}(s,x)=\rho(s+1,x)\) in order to chain steps soundly.

- **Current practice (as in code today)**: the packed witness includes \(\rho\), \(\rho_{\text{next}}\), and \(Q\) as committed columns in the
  recursion dense commitment. Stage 2 runs `GtShift` to enforce the global shift relation \(\rho_{\text{next}}(s,x)=\rho(s+1,x)\), and runs
  `GtExpClaimReduction` to move the GT-exp openings onto the shared Stage-2 point \(r_x\) used by the rest of the system.
- **Potential optimization (not yet enabled)**: eliminate the committed \(\rho_{\text{next}}\) column and instead treat
  \(\rho_{\text{next}}(r^\*)\) as a purely virtual claim derived from \(\rho\) via `GtShift`. This would reduce committed witness size, but
  would change proof semantics and must be versioned.

Concretely (matching Dory’s `DoryVerifierState::verify_final`), the three pairs are:

- \(p1 = (E1_{\text{final}} + d\cdot g1_0,\; E2_{\text{final}} + d^{-1}\cdot g2_0)\)
- \(p2 = (h1,\; (-\gamma)\cdot(E2_{\text{acc}} + (d^{-1}\cdot s1_{\text{acc}})\cdot g2_0))\)
- \(p3 = ((-\gamma^{-1})\cdot(E1_{\text{acc}} + (d\cdot s2_{\text{acc}})\cdot g1_0) + d^2\cdot E1_{\text{init}},\; h2)\)

where:
- \(E1_{\text{init}} = \text{vmv.e1}\) (from the Dory proof),
- \(E1_{\text{acc}},E2_{\text{acc}}\) are the G1/G2 accumulators after all reduce-and-fold rounds,
- \(E1_{\text{final}},E2_{\text{final}}\) are the final message elements from the Dory proof, and
- \(d,\gamma,s1_{\text{acc}},s2_{\text{acc}}\) are derived by the outside verifier from the transcript / evaluation point.

### 1.5 Field Choice

All SNARK arithmetic is over $\mathbb{F}_q$ (BN254 base field), which equals the Grumpkin scalar field. This choice is dictated by:
- GT witnesses produce $\mathbb{F}_q$ elements (from Fq12 representation)
- Hyrax PCS operates over Grumpkin, whose scalar field is $\mathbb{F}_q$

### 1.6 Public API (Recursion Artifact)

The public wrapper API lives in `jolt-core/src/zkvm/recursion/api.rs` and exposes:

- `prove_recursion(...) -> RecursionArtifact`
- `verify_recursion(..., recursion: &RecursionArtifact) -> Result<()>`

The standalone artifact is serialized separately from the base `JoltProof`:

```rust
pub struct RecursionArtifact<FS: Transcript> {
    /// Required by the verifier (rejects if `None`).
    pub stage8_combine_hint: Option<Fq12>,
    /// External pairing boundary (3 pairing input pairs + expected GT rhs).
    pub pairing_boundary: PairingBoundary,
    /// Hints used for verifier-side plan derivation (guest recomputes without trusting).
    pub non_input_base_hints: NonInputBaseHints,
    /// The recursion SNARK proof (sumchecks + Hyrax opening).
    pub proof: RecursionProof<Fq, FS, HyraxPCS>,
}
```

**Trust model**:
- The verifier must treat `pairing_boundary` and `non_input_base_hints` as *hints* (untrusted data). With wiring/boundary constraints enabled
  (default), Stage 2 binds these values to the verifier-derived AST. The outside verifier then performs the final pairing check against
  `pairing_boundary.rhs`.
- `stage8_combine_hint` is also a hint, but is currently **required** for verification (it is `Option` for serialization compatibility).

---

## 2. Constraint Sumchecks (Stage 1 + Stage 2)

Stage 1 runs the packed GT exponentiation sumcheck (`SumcheckId::GtExp`). Stage 2 runs the remaining sumchecks (shift checks, claim reduction, GT mul, G1/G2 scalar mul, G1/G2 add, wiring/boundary constraints) batched together. This section documents the gadget-level constraints independent of which stage they run in.

### 2.1 Ring Switching & Quotient Technique

$\mathbb{G}_T$ elements are represented as $\mathbb{F}_{q^{12}} = \mathbb{F}_q[X]/(p(X))$ where $p(X)$ is an irreducible polynomial of degree 12.

**Key insight**: For $a, b, c \in \mathbb{G}_T$, the equation $a \cdot b = c$ holds iff there exists quotient $Q$ such that:
$$a(X) \cdot b(X) = c(X) + Q(X) \cdot p(X)$$

On the Boolean hypercube $\{0,1\}^4$ (viewing elements as 4-variate polynomials):
$$a(x) \cdot b(x) - c(x) - Q(x) \cdot g(x) = 0$$

where $g(x)$ is the MLE of $p$ on the hypercube.

This transforms high-degree $\mathbb{F}_{q^{12}}$ operations into low-degree constraints over $\mathbb{F}_q$ by introducing the quotient as auxiliary witness.

### 2.2 GT Exponentiation (Stage 1)

Computes $b = a^k$ using iterative exponentiation.

#### Inputs / Outputs

| Role | Symbol | Description |
|------|--------|-------------|
| Witness (Input) | $a \in \mathbb{G}_T$ | Base (as Fq12) |
| Public Input | $k \in \mathbb{F}_r$ | Exponent |
| Witness (Output) | $b \in \mathbb{G}_T$ | Result $b = a^k$ |

#### Witness

| Symbol | Description |
|--------|-------------|
| $(b_0, \ldots, b_{t-1})$ | Binary representation of $k$ |
| $\rho_0, \ldots, \rho_t$ | Intermediate values: $\rho_0 = 1$, $\rho_t = a^k$ |
| $Q_0, \ldots, Q_{t-1}$ | Quotient polynomials |

Recurrence:
$$\rho_{i+1} = \begin{cases} \rho_i^2 & \text{if } b_i = 0 \\ \rho_i^2 \cdot a & \text{if } b_i = 1 \end{cases}$$

#### Constraint

For each bit $i$:
$$C_i(x) = \rho_{i+1}(x) - \rho_i(x)^2 \cdot a(x)^{b_i} - Q_i(x) \cdot g(x) = 0$$

where $a(x)^{b_i} = 1 + (a(x) - 1) \cdot b_i$ (linearization).

#### Sum-Check

$$0 = \sum_{x \in \{0,1\}^4} \text{eq}(r_x, x) \cdot \sum_{i=0}^{t-1} \gamma^i \cdot C_i(x)$$

- 4 rounds (one per variable)
- Batching coefficient $\gamma$ combines all $t$ constraints

#### Output Claims

After final challenge $r_x'$:
- `RecursionBase(i)`: $\tilde{a}(r_x')$
- `RecursionRhoPrev(i)`: $\tilde{\rho}_i(r_x')$
- `RecursionRhoCurr(i)`: $\tilde{\rho}_{i+1}(r_x')$
- `RecursionQuotient(i)`: $\tilde{Q}_i(r_x')$

#### Packed Witness Structure

Instead of creating separate polynomials for each step, we pack the exponentiation trace into a single table over
step variables and element variables. The implementation uses **base-4 digits** (two bits per step), which reduces
the number of steps and keeps the constraint degree low.

| Symbol | Description | Layout |
|--------|-------------|--------|
| $\rho(s, x)$ | Intermediate values | $\rho[x \cdot 128 + s] = \rho_s[x]$ |
| $\rho_{\text{next}}(s, x)$ | Shifted intermediate values | $\rho_{\text{next}}[x \cdot 128 + s] = \rho_{s+1}[x]$ |
| $Q(s, x)$ | Quotient polynomials | $Q[x \cdot 128 + s] = Q_s[x]$ |
| $\text{digit\_lo}(s),\ \text{digit\_hi}(s)$ | Base-4 digit bits of $k$ | Replicated across $x$ |
| $\text{base}(x),\text{base}^2(x),\text{base}^3(x)$ | Base powers | Replicated across $s$ |

Where:
- $s \in \{0,1\}^7$ indexes the step (0 to 127)
- $x \in \{0,1\}^4$ indexes the field element (0 to 15)
- Layout formula: `index = x * 128 + s` (s in low bits)

#### Unified Constraint

Let \((u_s,v_s) = (\text{digit\_lo}(s),\text{digit\_hi}(s))\) encode a base-4 digit \(d_s \in \{0,1,2,3\}\).
Define weights:
\[
w_0=(1-u)(1-v),\; w_1=u(1-v),\; w_2=(1-u)v,\; w_3=uv
\]
and the selected base power:
\[
\text{base\_power}(x,s)=w_0 + w_1\cdot \text{base}(x) + w_2\cdot \text{base}^2(x) + w_3\cdot \text{base}^3(x).
\]

The packed transition constraint is:

$$C(s, x) = \rho_{\text{next}}(s, x) - \rho(s, x)^4 \cdot \text{base\_power}(x,s) - Q(s, x) \cdot g(x) = 0.$$

#### Two-Phase Sum-Check

$$0 = \sum_{s \in \{0,1\}^7} \sum_{x \in \{0,1\}^4} \text{eq}(r_s, s) \cdot \text{eq}(r_x, x) \cdot C(s, x).$$

- **Phase 1** (rounds 0-6): Bind step variables $s$
- **Phase 2** (rounds 7-10): Bind element variables $x$
- Total: 11 rounds, degree 7 (constraint degree 6 times multilinear eq)

#### Output Claims

After final challenges $(r_s^*, r_x^*)$:
- `VirtualPolynomial::gt_exp_rho(i)`: $\rho(r_s^*, r_x^*)$
- `VirtualPolynomial::gt_exp_rho_next(i)`: $\rho_{\text{next}}(r_s^*, r_x^*)$
- `VirtualPolynomial::gt_exp_quotient(i)`: $Q(r_s^*, r_x^*)$

The verifier computes \(\text{digit\_lo}(r_s^*)\), \(\text{digit\_hi}(r_s^*)\), and \(\text{base}(r_x^*)\), \(\text{base}^2(r_x^*)\), \(\text{base}^3(r_x^*)\)
directly from public inputs (scalar bits and base), so these are **not** emitted as openings/claims.

#### Public Polynomial Optimization

The `bit` (digit) and `base` polynomials are derived entirely from **public inputs** (the scalar exponent and base element), so the prover does not commit to them:

| Polynomial | Source | Verifier Action |
|------------|--------|-----------------|
| `digit_lo(s)`, `digit_hi(s)` | Scalar $k$ bits | Evaluate 7-variable MLE at $r_s^*$ |
| `base(x)`, `base²(x)`, `base³(x)` | Base $a \in \mathbb{G}_T$ | Evaluate 4-variable MLEs at $r_x^*$ |
| `g(x)` | Irreducible polynomial (constant) | Evaluate 4-variable MLE at $r_x^*$ |

This follows the same pattern as the irreducible polynomial `g(x)`, which is a known constant that the verifier evaluates directly. By recognizing these as public inputs:

- **5 → 3 polynomials** committed per GT exponentiation
- **40% reduction** in virtual claims for GT exp
- **No security impact** — verifier computes identical values from public data

The prover still uses these polynomials internally during sumcheck computation, but does not include them in the commitment.

#### Mathematical Correctness

**Theorem**: The packed representation maintains constraint satisfaction equivalence.

**Proof**: Let \((u_s,v_s)\) be the two bit selectors for the base-4 digit \(d_s \in \{0,1,2,3\}\).
For each step \(s \in [0, \text{num\_steps}-1] \subseteq [0,127]\) and element index \(x \in [0,15]\):

- By construction of packing, \(\rho(s,x)=\rho_s(x)\), \(\rho_{\text{next}}(s,x)=\rho_{s+1}(x)\), and \(Q(s,x)=Q_s(x)\).
- By construction of the weight selection, \(\text{base\_power}(x,s)=\text{base}(x)^{d_s}\).

Therefore the packed constraint \(C(s,x)=0\) is exactly the per-step base-4 transition constraint for the original trace,
and constraint satisfaction is preserved. □

**Security Analysis**:
- Soundness error: Unchanged at $\text{deg}/|\mathbb{F}| = 7/p \approx 2^{-252}$
- The two-phase sumcheck maintains the same security as checking all packed step constraints (up to 128 base-4 steps)
- Batching with $\gamma$ preserves zero-knowledge properties

#### Performance Benefits

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Polynomials per GT exp | 1,024 | 3 | 341.3× |
| Virtual claims | 1,024 | 3 | 341.3× |
| Proof size contribution | ~32KB | ~96B | ~333× |

---

### 2.2.1 GtShift: Packed GT exp internal shift check (Stage 2)

The packed GT exponentiation witness contains both `rho(s,x)` and `rho_next(s,x)` columns. The `GtExp` constraint enforces the *local* per-step
transition using these columns, but it does **not** by itself enforce that the prover’s `rho_next` column is globally consistent with `rho` across
all steps. The `GtShift` sumcheck enforces the global shift relation:
\[
\rho_{\text{next}}(s,x) = \rho(s+1,x).
\]

#### What `GtShift` proves (as implemented)

Let \(r^*=(r_s^*, r_x^*)\) be the opening point used by the Stage-1 `GtExp` sumcheck. Let \(v\) denote the opened value of the witness column
\(\rho_{\text{next}}\) at that point:
\[
v := \rho_{\text{next}}(r_s^*, r_x^*).
\]

The `GtShift` sumcheck proves that this opened value is consistent with the committed `rho` column shifted by one in the step index:

Let \(r^*=(r_s^*, r_x^*)\) be the opening point used by `GtExp`. Let \(v\) denote the claimed value:
\[
v := \rho_{\text{next}}(r_s^*, r_x^*).
\]

Then the shift sumcheck proves:
\[
v \;=\;
\sum_{s \in \{0,1\}^7} \sum_{x \in \{0,1\}^4}
\mathrm{EqPlusOne}(r_s^*, s)\cdot \mathrm{Eq}(r_x^*, x)\cdot \rho(s,x),
\]
where \(\mathrm{EqPlusOne}(r_s^*, s)\) is the multilinear selector for “\(s = r_s^* + 1\)” (with the out-of-range boundary contributing 0).

Where:
- `EqPlusOne(r_s*, s)` = 1 if `s = r_s* + 1`, 0 otherwise
- The sum evaluates to exactly `rho(r_s*+1, r_x*)` = `rho_next(r_s*, r_x*)`

#### Protocol positioning (Stage 1 → Stage 2)

```
Stage 1: Packed GT Constraint Sumcheck (`GtExp`)
  - Uses committed witness columns: rho, rho_next, quotient
  - Outputs openings at r*: rho(r*), rho_next(r*), quotient(r*)
  ↓
Stage 2: Shift sumcheck (`GtShift`)
  - Proves that rho_next(r_s*, r_x*) = rho(r_s*+1, r_x*)
  - 11 rounds, degree 3
  - Caches the corresponding rho opening under `SumcheckId::GtShift` for later reductions
  ↓
Stage 3: Prefix packing reduction + PCS opening
  - Reduces all Stage-2 virtual openings to one committed opening claim on the packed dense polynomial.
  - Proves that opening via Hyrax.
```

#### Accumulator communication (as implemented)

In the implementation:
- `GtExp` caches the virtual claim `VirtualPolynomial::gt_exp_rho_next(i)` at the `GtExp` opening point (under `SumcheckId::GtExp`).
- `GtShift` caches the corresponding `rho` opening `VirtualPolynomial::gt_exp_rho(i)` at the *GtShift* opening point (under `SumcheckId::GtShift`).

The verifier checks consistency by combining these cached openings with the appropriate Eq/EqPlusOne selector evaluations (see `jolt-core/src/zkvm/recursion/gt/shift.rs`).

#### Notes / future optimization

It is possible to remove the committed `rho_next` column entirely (since it is determined by `rho` via a shift), but doing so requires treating
`rho_next(r^*)` as a purely virtual claim derived from `rho` and adjusting the proof format. The current code keeps `rho_next` as a committed
witness column and uses `GtShift` to enforce its consistency with `rho`.

#### Benefits and Trade-offs

**Benefits**:
- Enforces the global packed-trace consistency relation \(\rho_{\text{next}}(s,x)=\rho(s+1,x)\) (required for soundness).
- Enables a future optimization where the committed `rho_next` column could be removed (with proof-format versioning).

**Trade-offs**:
- Adds 11 sumcheck rounds
- Slightly more complex verifier (+24 field ops per round)
- Additional proof elements (+11 field elements)

**Security**: Maintains the standard sumcheck soundness bound \(O(\text{rounds}/|\mathbb{F}|)\) (here, rounds = 11).

---

### 2.3 GT Multiplication (Stage 2)

Proves $c = a \cdot b$ for $a, b, c \in \mathbb{G}_T$.

#### Inputs / Outputs

| Role | Symbol | Description |
|------|--------|-------------|
| Witness (Input) | $a, b \in \mathbb{G}_T$ | Operands |
| Witness (Output) | $c \in \mathbb{G}_T$ | Result $c = a \cdot b$ |

#### Witness

| Symbol | Description |
|--------|-------------|
| $Q$ | Quotient polynomial: $Q(x) = \frac{a(x) \cdot b(x) - c(x)}{g(x)}$ |

#### Constraint

$$C(x) = a(x) \cdot b(x) - c(x) - Q(x) \cdot g(x) = 0$$

#### Sum-Check

For $m$ multiplication constraints:
$$0 = \sum_{x \in \{0,1\}^4} \text{eq}(r_x, x) \cdot \sum_{i=0}^{m-1} \gamma^i \cdot C_i(x)$$

#### Output Claims

- `VirtualPolynomial::gt_mul_lhs(i)`: $\tilde{a}_i(r_x')$
- `VirtualPolynomial::gt_mul_rhs(i)`: $\tilde{b}_i(r_x')$
- `VirtualPolynomial::gt_mul_result(i)`: $\tilde{c}_i(r_x')$
- `VirtualPolynomial::gt_mul_quotient(i)`: $\tilde{Q}_i(r_x')$

---

### 2.4 G1 Scalar Multiplication (Stage 2)

Proves $Q = [k]P$ using double-and-add.

#### Inputs / Outputs

| Role | Symbol | Description |
|------|--------|-------------|
| Witness/Public | $P \in \mathbb{G}_1$ | Base point $(x_P, y_P)$ |
| Public Input | $k \in \mathbb{F}_r$ | Scalar with bits $(b_0, \ldots, b_{n-1})$ |
| Witness (Output) | $Q \in \mathbb{G}_1$ | Result $Q = [k]P$ |

#### Witness

Execution trace for a 256-bit scalar (so $n=256$ rows). We encode infinity as affine
coordinates $(0,0)$ plus an indicator bit.

| Column | Description |
|--------|-------------|
| $(x_A, y_A)$ | Accumulator at start of iteration: $A_i$ (or $(0,0)$ if $A_i=\mathcal{O}$) |
| $(x_T, y_T)$ | Doubled point: $T_i = [2]A_i$ (or $(0,0)$ if $T_i=\mathcal{O}$) |
| $(x_{A\_\text{next}}, y_{A\_\text{next}})$ | Next accumulator: $A_{i+1}$ (stored as a shifted “next” table) |
| $\text{ind}_A$ | 1 if $A_i=\mathcal{O}$, else 0 |
| $\text{ind}_T$ | 1 if $T_i=\mathcal{O}$, else 0 |

Each row $i$ implements:
$$A_{i+1} = T_i + b_i \cdot P$$

**Bit handling**: the scalar bits $b_i$ are treated as **public inputs** derived from $k$; the
verifier recomputes $b(r^\*)$ from $k$ and does not require a committed/opened bit polynomial.

#### Constraints

**C1 (Doubling x)**: $4y_A^2(x_T + 2x_A) - 9x_A^4 = 0$

**C2 (Doubling y)**: $3x_A^2(x_T - x_A) + 2y_A(y_T + y_A) = 0$

**C3 (Conditional add x)**: bit-dependent conditional addition with a special case for $T=\mathcal{O}$:

$$
\begin{aligned}
0
=\;&(1-b)\cdot(x_{A\_\text{next}}-x_T) \\
&+ b\cdot \text{ind}_T \cdot (x_{A\_\text{next}}-x_P) \\
&+ b\cdot(1-\text{ind}_T)\cdot\Big[(x_{A\_\text{next}}+x_T+x_P)\cdot(x_P-x_T)^2-(y_P-y_T)^2\Big]
\end{aligned}
$$

**C4 (Conditional add y)**:

$$
\begin{aligned}
0
=\;&(1-b)\cdot(y_{A\_\text{next}}-y_T) \\
&+ b\cdot \text{ind}_T \cdot (y_{A\_\text{next}}-y_P) \\
&+ b\cdot(1-\text{ind}_T)\cdot\Big[(y_{A\_\text{next}}+y_T)\cdot(x_P-x_T)-(y_P-y_T)\cdot(x_T-x_{A\_\text{next}})\Big]
\end{aligned}
$$

**C5 (Doubling preserves infinity)**: if $A_i=\mathcal{O}$ then $T_i=\mathcal{O}$:

$$\text{ind}_A \cdot (1-\text{ind}_T) = 0$$

**C6 (Infinity encoding for $T$)**: if $\text{ind}_T=1$ then $(x_T,y_T)=(0,0)$ (field-independent):

$$\text{ind}_T \cdot x_T = 0 \qquad\text{and}\qquad \text{ind}_T \cdot y_T = 0$$

| Constraint | Degree |
|------------|--------|
| C1 | 4 |
| C2 | 3 |
| C3 | 5 |
| C4 | 4 |
| C5 | 2 |
| C6 | 2 |

#### Sum-Check

The scalar-mul trace is a native **8-variable** MLE over the 256 steps. For each scalar-mul instance, constraints are batched with $\delta$,
and multiple instances are batched with $\gamma$:

$$
0 = \sum_{x \in \{0,1\}^{8}} \text{eq}(r_x, x)\cdot \sum_{i}\gamma^i\cdot\Big(\sum_{j}\delta^j\cdot C_{i,j}(x)\Big)
$$

- 8 rounds (one per step variable). In the Stage-2 batched sumcheck, shorter instances are suffix-aligned, so their \(r_x\) point is the
  suffix of the overall Stage-2 challenge vector.
- Degree 6 overall (max constraint degree 5, times $\text{eq}(r_x,x)$ which is multilinear)

#### Output Claims

- `VirtualPolynomial::g1_scalar_mul_xa(i)`, `VirtualPolynomial::g1_scalar_mul_ya(i)` - accumulator coordinates
- `VirtualPolynomial::g1_scalar_mul_xt(i)`, `VirtualPolynomial::g1_scalar_mul_yt(i)` - doubled point coordinates
- `VirtualPolynomial::g1_scalar_mul_xa_next(i)`, `VirtualPolynomial::g1_scalar_mul_ya_next(i)` - next accumulator
- `VirtualPolynomial::g1_scalar_mul_t_indicator(i)` - 1 if T = O (infinity), 0 otherwise
- `VirtualPolynomial::g1_scalar_mul_a_indicator(i)` - 1 if A = O (infinity), 0 otherwise

---

### 2.5 G2 Scalar Multiplication (Stage 2)

Proves $R = [k]Q$ for $Q \in \mathbb{G}_2$ using the same 256-step MSB-first double-and-add trace
as G1, but with coordinates in the quadratic extension field $\mathbb{F}_{q^2}$.

#### Field representation ($\mathbb{F}_{q^2}$)

We use the standard BN254 quadratic extension:
$$\mathbb{F}_{q^2} = \mathbb{F}_q[u]/(u^2 + 1)$$
so each element is represented as $a = a_0 + a_1 \cdot u$ with $a_0,a_1 \in \mathbb{F}_q$ and
$u^2 = -1$.

Since the recursion SNARK arithmetic is over $\mathbb{F}_q$ (BN254 base field), we **split every**
$\mathbb{F}_{q^2}$ coordinate into its $(c0,c1)$ components in $\mathbb{F}_q$ and enforce all
constraints component-wise.

#### Inputs / Outputs

| Role | Symbol | Description |
|------|--------|-------------|
| Witness/Public | $Q \in \mathbb{G}_2$ | Base point $(x_Q, y_Q) \in \mathbb{F}_{q^2}^2$ |
| Public Input | $k \in \mathbb{F}_r$ | Scalar (256-bit, MSB-first bits $b_0,\ldots,b_{255}$) |
| Witness (Output) | $R \in \mathbb{G}_2$ | Result $R = [k]Q$ |

#### Witness

As in G1, we maintain a 256-row trace with infinity encoded as $(0,0)$ plus an indicator bit.
All $\mathbb{F}_{q^2}$ values are split into $(c0,c1)$ over $\mathbb{F}_q$.

| Column | Description |
|--------|-------------|
| $(x_A, y_A)$ | Accumulator $A_i$ (Fq2; split into c0/c1 over Fq) |
| $(x_T, y_T)$ | Doubled point $T_i = [2]A_i$ (Fq2; split into c0/c1 over Fq) |
| $(x_{A\_\text{next}}, y_{A\_\text{next}})$ | Next accumulator $A_{i+1}$ (shifted “next” table) |
| $\text{ind}_A$ | 1 if $A_i=\mathcal{O}$, else 0 (in $\mathbb{F}_q$) |
| $\text{ind}_T$ | 1 if $T_i=\mathcal{O}$, else 0 (in $\mathbb{F}_q$) |

Each row $i$ implements:
$$A_{i+1} = T_i + b_i \cdot Q$$

**Bit handling**: as in G1, bits are treated as **public inputs** derived from $k$; the verifier
recomputes $b(r^\*)$ from $k$.

#### Constraints (conceptual, in $\mathbb{F}_{q^2}$)

The affine short-Weierstrass group laws (denominator-free) are identical in form to G1 and apply
over $\mathbb{F}_{q^2}$:

- **C1 (Doubling x)**: $4y_A^2(x_T + 2x_A) - 9x_A^4 = 0$
- **C2 (Doubling y)**: $3x_A^2(x_T - x_A) + 2y_A(y_T + y_A) = 0$
- **C3 (Conditional add x)**: same bit-dependent form as G1, but over $\mathbb{F}_{q^2}$
- **C4 (Conditional add y)**: same bit-dependent form as G1, but over $\mathbb{F}_{q^2}$

Infinity handling is enforced in $\mathbb{F}_q$:

- **C5 (Doubling preserves infinity)**:
  $$\text{ind}_A \cdot (1-\text{ind}_T) = 0$$
- **C6 (Infinity encoding for $T$)**: if $\text{ind}_T=1$ then $(x_T,y_T)=(0,0)$ in $\mathbb{F}_{q^2}$.
  Implemented as four base-field constraints:
  $$\text{ind}_T \cdot x_{T,c0}=0,\;\text{ind}_T \cdot x_{T,c1}=0,\;\text{ind}_T \cdot y_{T,c0}=0,\;\text{ind}_T \cdot y_{T,c1}=0.$$

#### Sum-Check (implemented batching layout)

The sumcheck runs over **8** step variables (256 steps).
Because each $\mathbb{F}_{q^2}$ constraint contributes two $\mathbb{F}_q$ equations (c0 and c1),
the implementation batches **13** $\mathbb{F}_q$ constraint terms per instance:

- C1.c0, C1.c1
- C2.c0, C2.c1
- C3.c0, C3.c1
- C4.c0, C4.c1
- C5
- C6.x\_c0, C6.x\_c1, C6.y\_c0, C6.y\_c1

These are batched with $\delta$, and multiple instances are batched with $\gamma$:

$$
0 = \sum_{x \in \{0,1\}^{8}} \text{eq}(r_x, x)\cdot \sum_{i}\gamma^i\cdot\Big(\sum_{j}\delta^j\cdot C_{i,j}(x)\Big)
$$

- 8 rounds (suffix-aligned in the Stage-2 batched sumcheck)
- Degree 6 overall (max term degree 5, times $\text{eq}(r_x,x)$)

#### Output Claims

All opened values are over $\mathbb{F}_q$ (components of $\mathbb{F}_{q^2}$):

- `VirtualPolynomial::g2_scalar_mul_xa_c0(i)`, `VirtualPolynomial::g2_scalar_mul_xa_c1(i)`
- `VirtualPolynomial::g2_scalar_mul_ya_c0(i)`, `VirtualPolynomial::g2_scalar_mul_ya_c1(i)`
- `VirtualPolynomial::g2_scalar_mul_xt_c0(i)`, `VirtualPolynomial::g2_scalar_mul_xt_c1(i)`
- `VirtualPolynomial::g2_scalar_mul_yt_c0(i)`, `VirtualPolynomial::g2_scalar_mul_yt_c1(i)`
- `VirtualPolynomial::g2_scalar_mul_xa_next_c0(i)`, `VirtualPolynomial::g2_scalar_mul_xa_next_c1(i)`
- `VirtualPolynomial::g2_scalar_mul_ya_next_c0(i)`, `VirtualPolynomial::g2_scalar_mul_ya_next_c1(i)`
- `VirtualPolynomial::g2_scalar_mul_t_indicator(i)`, `VirtualPolynomial::g2_scalar_mul_a_indicator(i)`

---

### 2.6 Shift checks for scalar-mul traces (Stage 2)

Scalar multiplication witnesses include both an “accumulator” column \(A_i\) and an explicitly shifted “next accumulator” column
\(A'_{i}=A_{i+1}\) (named `A_next` / `x_a_next` / `y_a_next` in the code). Without an explicit shift constraint, a prover could
choose unrelated per-step states across rows while still satisfying each step’s local group-law constraints.

We therefore run dedicated shift sumchecks:
- `SumcheckId::ShiftG1ScalarMul` over `(x_a, x_a_next)` and `(y_a, y_a_next)`
- `SumcheckId::ShiftG2ScalarMul` over the four coordinate components
  `(x_a_c0, x_a_next_c0)`, `(x_a_c1, x_a_next_c1)`, `(y_a_c0, y_a_next_c0)`, `(y_a_c1, y_a_next_c1)`

#### Precise relation

Let the step index be \(i \in \{0,\dots,255\}\). The witness is constructed as:
- \(A(i) = A_i\) for \(i=0..255\)
- \(A_{\text{next}}(i) = A_{i+1}\) for \(i=0..255\) (so \(A_{\text{next}}(255)=A_{256}\) is the *final* result)

The shift relation we need is:
\[
\forall i \in \{0,\dots,254\}:\quad A_{\text{next}}(i)=A(i+1).
\]

#### Randomized check (native 8-var step domain)

In the implementation (`jolt-core/src/zkvm/recursion/g1/shift.rs`), we treat the step index as an 8-bit Boolean cube and prove a randomized
linear identity that enforces the one-step shift over all indices \(0..254\), while excluding the terminal boundary \(A_{\text{next}}(255)\):
\[
\sum_{s} \mathrm{Eq}(r^*, s)\cdot (1-\mathrm{Eq}(s,2^8\!-\!1))\cdot A_{\text{next}}(s)
\;=\;
\sum_{s} \mathrm{Eq}(r^*, s-1)\cdot A(s).
\]

This shift sumcheck is defined over the **native 8-var** scalar-mul trace polynomials. In the Stage-2 batched sumcheck, it is suffix-aligned
via `round_offset` so it can be batched with other instances that have more rounds.

---

### 2.7 Multi-Miller loop (BN254 pairing Miller loop)

**Implementation status (current code)**: pairing recursion is **experimental** and gated behind `feature = "experimental-pairing-recursion"`.
The gadgets exist (`jolt-core/src/zkvm/recursion/pairing/*`), and the Stage-2 prover/verifier contain feature-gated hooks, but the **streaming**
recursion pipeline does not currently include pairing-native stores by default:
- Prover: `RecursionProver::prove_stage2` has a TODO to wire pairing recursion native stores into the pipeline (`jolt-core/src/zkvm/recursion/prover.rs`, ~L1260–L1265).
- Planner: `witness_generation::plan_constraint_system` currently does not claim pairing support in the streaming plan (`jolt-core/src/zkvm/recursion/witness_generation.rs`, “Pairing (experimental)” section).

When enabled end-to-end, the intended Stage-2 integration is:
- `SumcheckId::MultiMillerLoop` (`jolt-core/src/zkvm/recursion/pairing/multi_miller_loop.rs`)
- `SumcheckId::ShiftMultiMillerLoop` (`jolt-core/src/zkvm/recursion/pairing/shift.rs`)

This gadget is intended to prove correctness of the **Miller-loop** part of a BN254 pairing computation:
\[
f_{\text{out}} \;=\; \prod_{s=0}^{S-1} \Big(f_s^{[2]} \cdot \ell_s(P)\Big)^{\mathbf{1}_{\text{double}}(s)}
\cdot \Big(f_s \cdot \ell_s(P)\Big)^{\mathbf{1}_{\text{add}}(s)}
\]
where each step \(s\) is either a tangent (doubling) step or a chord (addition) step, and \(\ell_s(P)\) is the line function evaluated
at the G1 input point \(P\). (Final exponentiation and/or the final multi-pairing equality can still remain outside the SNARK boundary.)

#### Packed trace layout (11-variable domain)

We use the same packed layout as the packed GT exponentiation:
- step variables \(s \in \{0,1\}^7\) (up to 128 steps),
- element variables \(x \in \{0,1\}^4\) (16 base-field evaluations representing an \(F_{q^{12}}\) element via an MLE),
- packed index: `idx = x * 128 + s` (step in the low bits).

Each operation instance corresponds to one Miller-loop trace (typically one \((P,Q)\) pair), represented as 11-var MLE columns.

#### Per-step constraints (local semantics)

At each packed point \((s,x)\), the constraint polynomial enforces:

1. **Branch bits** (`is_double`, `is_add`) are boolean and mutually exclusive.
2. **G2 affine update** (over \(F_{q^2}\), split into base-field components):
   - doubling slope: \(2y\lambda = 3x^2\),
   - addition slope: \(\lambda(x_Q-x_T) = y_Q-y_T\), plus an `inv_dx` witness enforcing \((x_Q-x_T)^{-1}\) in the add branch,
   - affine formulas: \(x'=\lambda^2-x_T-x_{\text{op}}\), \(y'=\lambda(x_T-x')-y_T\).
3. **Line coefficients** \((c0,c1,c2)\in F_{q^2}\) for the tangent/chord line, using the BN254 `TwistType::D` conventions.
4. **Line evaluation embedding**: the line is embedded as a sparse-034 \(F_{q^{12}}\) element and evaluated via 6 selector polynomials.
5. **Accumulator update in \(F_{q^{12}}\)** via ring-switching:
   \[
   a(x)\cdot b(x) - c(x) - Q(x)\cdot g(x) = 0
   \]
   where \(a\) is either \(f^2\) (double) or \(f\) (add), \(b\) is the embedded line value, \(c\) is \(f_{\text{next}}\), and
   \(g\) is the fixed tower reduction polynomial MLE.

These per-step constraints are **sound and complete** for the *local* step semantics, assuming all “public” polynomials described below
are fixed correctly (see caveats).

#### Missing global consistency constraints (required for end-to-end soundness)

Local step correctness alone does **not** imply the trace is a single coherent Miller-loop computation. We must additionally enforce:

1. **Shift (step-to-step chaining)**: for each relevant column \(A\),
   \[
   \forall s<127,\forall x:\quad A_{\text{next}}(s,x) = A(s+1,x)
   \]
   This prevents a prover from choosing unrelated per-step states. We implement this as a dedicated shift sumcheck:
   - `SumcheckId::ShiftMultiMillerLoop` (implemented in `jolt-core/src/zkvm/recursion/pairing/shift.rs`),
   - applied at least to `f`/`f_next` and the four G2 state components `t_*`/`t_*_next`.

2. **Boundary conditions**:
   - initial state \(f(0)=1\) and \(T(0)=Q\),
   - correct step schedule (derived from `Bn254Config::ATE_LOOP_COUNT` plus the two final Frobenius additions),
   - extraction of the final output slice \(f(\text{last},x)\) as the claimed Miller-loop output.

3. **Public constants vs witness values**:
   - `g(x)` is a fixed, verifier-known polynomial (`get_g_mle()`), and must not be prover-chosen,
   - the sparse-034 selector polynomials are fixed constants (basis vectors under `fq12_to_multilinear_evals`) and must not be prover-chosen.
   As with packed GT exponentiation, the intended design is that the verifier computes these values directly from public definitions; if they are
   carried as witness columns for convenience, we must add explicit constraints that they match their public definitions.

#### Notes / remaining TODOs

The Stage 2 MultiMillerLoop gadgets enforce *local* step semantics and (via `ShiftMultiMillerLoop`) enforce step-to-step chaining for the
`*_next` columns. Boundary conditions (initialization/finalization) and “public constant” treatment (e.g. `g(x)` / selector columns) should
be documented/locked down alongside the witness generator.

### 2.5.1 G1/G2 Addition (NEW)

Dory verification performs many explicit group additions in G1 and G2 (e.g., updates like
\(e1 \leftarrow e1 + \alpha \cdot X\), \(e2 \leftarrow e2 + \beta^{-1}\cdot Y\)).
In Dory’s AST these appear as `G1Add` / `G2Add` nodes, and they must be proven inside the recursion SNARK
because they affect the final pairing inputs.

We add two new Stage 2 sumchecks:

- **G1Add sumcheck**: proves that for each add instance, the output point equals \(P+Q\) in \(\mathbb{G}_1\), with correct
  handling of infinity, doubling, and inverse cases.
- **G2Add sumcheck**: same, but for \(\mathbb{G}_2\) over \(\mathbb{F}_{q^2}\), implemented by splitting \(\mathbb{F}_{q^2}\)
  coordinates into (c0,c1) components in \(\mathbb{F}_q\) and enforcing constraints component-wise.

The constraint system below is the **authoritative** description of what is implemented in:

- `jolt-core/src/zkvm/recursion/g1/addition.rs`
- `jolt-core/src/zkvm/recursion/g2/addition.rs`

#### Goal (per instance)

Given witness polynomials for points \(P,Q,R\), prove:
\[
R = P + Q
\]
in the appropriate elliptic-curve group, including all exceptional cases.

#### Point representation (infinity encoding)

We encode points as affine coordinates plus an indicator bit:
\[
P=(x_P,y_P,\mathrm{ind}_P),\quad Q=(x_Q,y_Q,\mathrm{ind}_Q),\quad R=(x_R,y_R,\mathrm{ind}_R)
\]
where \(\mathrm{ind}=1\) indicates the point at infinity \(\mathcal{O}\).

Infinity is encoded field-independently by enforcing:
\[
\mathrm{ind}=1 \Rightarrow x=0 \ \wedge\ y=0
\]
via constraints \(\mathrm{ind}\cdot x = 0\) and \(\mathrm{ind}\cdot y = 0\).

We also enforce booleanity of all indicator / branch bits: \(b(1-b)=0\).

#### Auxiliary witnesses (division-free)

To avoid explicit inversion/division in-circuit, each addition instance includes auxiliary witnesses:

- \(\lambda\): the slope used in affine add/double formulas
- \(\mathrm{inv\_dx}\): inverse of \(\Delta x = x_Q - x_P\) in the generic-add branch (otherwise arbitrary)
- \(b_d\): `is_double` (1 iff the instance is treated as doubling in the finite case)
- \(b_i\): `is_inverse` (1 iff the instance is treated as the inverse case \(P=-Q\) in the finite case)

Define:
\[
\Delta x = x_Q - x_P,\quad \Delta y = y_Q - y_P,\quad
S_\text{finite} = (1-\mathrm{ind}_P)(1-\mathrm{ind}_Q).
\]

All of the above are witness values for a single add instance. In the current recursion constraint system, the `G1Add` and `G2Add` sumchecks use
`num_vars = 0`: there are no \(x\)-variables to range over, and the sumcheck checks the (batched) algebraic constraints at a single point.

More generally, different constraint families use different native arities (e.g. GT operations are 4-var, scalar-mul traces are 8-var, packed GT exp is
11-var). In the Stage-2 `BatchedSumcheck`, instances with fewer rounds are **suffix-aligned** via `round_offset` so they can be verified together.

#### Constraints (G1Add) over \(\mathbb{F}_q\)

All constraints below are identities in \(\mathbb{F}_q\). The finite-case constraints are gated by \(S_\text{finite}\),
and the “Q is infinity but P is not” case is gated by \(\mathrm{ind}_Q(1-\mathrm{ind}_P)\).

1. **Indicator booleanity**:
\[
\mathrm{ind}_P(1-\mathrm{ind}_P)=0,\quad
\mathrm{ind}_Q(1-\mathrm{ind}_Q)=0,\quad
\mathrm{ind}_R(1-\mathrm{ind}_R)=0.
\]

2. **Infinity encoding** (if \(\mathrm{ind}=1\) then \((x,y)=(0,0)\)):
\[
\mathrm{ind}_P x_P=0,\ \mathrm{ind}_P y_P=0,\quad
\mathrm{ind}_Q x_Q=0,\ \mathrm{ind}_Q y_Q=0,\quad
\mathrm{ind}_R x_R=0,\ \mathrm{ind}_R y_R=0.
\]

3. **Identity handling**:

- If \(P=\mathcal{O}\) then \(R=Q\):
\[
\mathrm{ind}_P(x_R-x_Q)=0,\quad
\mathrm{ind}_P(y_R-y_Q)=0,\quad
\mathrm{ind}_P(\mathrm{ind}_R-\mathrm{ind}_Q)=0.
\]

- If \(Q=\mathcal{O}\) and \(P\neq\mathcal{O}\) then \(R=P\):
\[
\mathrm{ind}_Q(1-\mathrm{ind}_P)(x_R-x_P)=0,\quad
\mathrm{ind}_Q(1-\mathrm{ind}_P)(y_R-y_P)=0,\quad
\mathrm{ind}_Q(1-\mathrm{ind}_P)(\mathrm{ind}_R-\mathrm{ind}_P)=0.
\]

4. **Branch-bit booleanity (finite case)**:
\[
S_\text{finite}\cdot b_d(1-b_d)=0,\quad
S_\text{finite}\cdot b_i(1-b_i)=0.
\]

5. **Branch selection / inverse-as-witness**:

Let \(b_\text{add} = 1-b_d-b_i\). We enforce:
\[
S_\text{finite}\cdot b_\text{add}\cdot (1-\mathrm{inv\_dx}\cdot\Delta x)=0.
\]
This has two effects:
- if \(\Delta x\neq 0\) then \(\mathrm{inv\_dx}=\Delta x^{-1}\) is forced in the add-branch,
- if \(\Delta x=0\) then \(b_\text{add}=0\), so we must be in a special case (doubling or inverse).

6. **Doubling / inverse declarations (finite case)**:

- If \(b_d=1\) then \(P=Q\):
\[
S_\text{finite}\cdot b_d\cdot\Delta x = 0,\quad
S_\text{finite}\cdot b_d\cdot(y_Q-y_P)=0.
\]

- If \(b_i=1\) then \(P=-Q\):
\[
S_\text{finite}\cdot b_i\cdot\Delta x = 0,\quad
S_\text{finite}\cdot b_i\cdot(y_Q+y_P)=0.
\]

7. **Slope equation (finite case)**:

We enforce a single linearized slope constraint that covers both add and double branches:
\[
S_\text{finite}\cdot\Big(
b_\text{add}\cdot(\Delta x\cdot\lambda - \Delta y)
\ +\ b_d\cdot(2y_P\cdot\lambda - 3x_P^2)
\Big)=0.
\]
If \(b_i=1\), then \(b_\text{add}=0\) and \(b_d=0\), so the slope constraint vanishes as intended.

8. **Result encoding and affine formulas (finite case)**:

- Inverse case: \(b_i=1 \Rightarrow R=\mathcal{O}\):
\[
S_\text{finite}\cdot b_i\cdot(1-\mathrm{ind}_R)=0.
\]

- Non-inverse case: \(b_i=0 \Rightarrow \mathrm{ind}_R=0\) and affine add/double formulas hold:
\[
S_\text{finite}\cdot (1-b_i)\cdot \mathrm{ind}_R = 0,
\]
\[
S_\text{finite}\cdot (1-b_i)\cdot\Big(x_R - (\lambda^2 - x_P - x_Q)\Big)=0,
\]
\[
S_\text{finite}\cdot (1-b_i)\cdot\Big(y_R - (\lambda(x_P-x_R)-y_P)\Big)=0.
\]

#### Batching / sumcheck statement (G1Add)

Within an instance, we batch the above constraints with \(\delta\) into a single polynomial:
\[
C(P,Q,R,\lambda,\mathrm{inv\_dx},b_d,b_i) \;=\; \sum_{j} \delta^j \cdot C_j.
\]
Across instances, we batch with \(\gamma\). The sumcheck proves:
\[
0 = \sum_{i}\gamma^i\cdot C_i.
\]

- Rounds: 0 (`num_vars = 0` in the current implementation)
- Degree bound: 6

*Note:* In the Stage-2 `BatchedSumcheck`, this 0-round instance is suffix-aligned via `round_offset`.

#### Output claims (ports)

At the (empty) evaluation point, Stage 2 caches virtual openings for the add ports/auxiliaries using `VirtualPolynomial::g1_add_*` helpers:

- `VirtualPolynomial::g1_add_xp(i)`, `VirtualPolynomial::g1_add_yp(i)`, `VirtualPolynomial::g1_add_p_indicator(i)`
- `VirtualPolynomial::g1_add_xq(i)`, `VirtualPolynomial::g1_add_yq(i)`, `VirtualPolynomial::g1_add_q_indicator(i)`
- `VirtualPolynomial::g1_add_xr(i)`, `VirtualPolynomial::g1_add_yr(i)`, `VirtualPolynomial::g1_add_r_indicator(i)`
- `VirtualPolynomial::g1_add_lambda(i)`, `VirtualPolynomial::g1_add_inv_delta_x(i)`
- `VirtualPolynomial::g1_add_is_double(i)`, `VirtualPolynomial::g1_add_is_inverse(i)`

These are the values Stage 2 wiring uses to connect add nodes to the rest of the AST DAG.

#### Constraints (G2Add) over \(\mathbb{F}_{q^2}\) implemented over \(\mathbb{F}_q\)

Conceptually, the G2 add constraint system is **identical** to G1, but with:

- \(x_\bullet,y_\bullet,\lambda,\mathrm{inv\_dx}\in\mathbb{F}_{q^2}\),
- \(\mathrm{ind}_\bullet,b_d,b_i\in\mathbb{F}_q\).

We use the BN254 quadratic extension:
\[
\mathbb{F}_{q^2} = \mathbb{F}_q[u]/(u^2+1),
\quad a = a_0 + a_1 u,
\]
and enforce all \(\mathbb{F}_{q^2}\) equations component-wise over \(\mathbb{F}_q\).

Concretely, the implementation in `g2/addition.rs` enforces the following \(\mathbb{F}_q\) constraints
(where each \(\mathbb{F}_{q^2}\) value is split into \((c0,c1)\)):

1. **Indicator booleanity**: \(\mathrm{ind}_P(1-\mathrm{ind}_P)=0\), \(\mathrm{ind}_Q(1-\mathrm{ind}_Q)=0\), \(\mathrm{ind}_R(1-\mathrm{ind}_R)=0\).
2. **Infinity encoding**: for each of the 12 base-field coordinates \(x_{P,c0},x_{P,c1},y_{P,c0},y_{P,c1},\ldots\) we enforce \(\mathrm{ind}\cdot \text{coord}=0\).
3. **Identity handling**:
   - If \(P=\mathcal{O}\), then \(R=Q\) component-wise and \(\mathrm{ind}_R=\mathrm{ind}_Q\).
   - If \(Q=\mathcal{O}\) and \(P\neq\mathcal{O}\), then \(R=P\) component-wise and \(\mathrm{ind}_R=\mathrm{ind}_P\).
4. **Branch-bit booleanity (finite case)**: \(S_\text{finite}\cdot b_d(1-b_d)=0\), \(S_\text{finite}\cdot b_i(1-b_i)=0\).
5. **Branch selection / inverse-as-witness (finite case)**:
   - Let \(\Delta x = x_Q-x_P \in \mathbb{F}_{q^2}\). In the add-branch \(b_\text{add}=1-b_d-b_i\), enforce:
     \[
     \mathrm{inv\_dx}\cdot\Delta x = 1 \in \mathbb{F}_{q^2}
     \]
     which becomes two \(\mathbb{F}_q\) constraints:
     \[
     (\mathrm{inv\_dx}\cdot\Delta x)_{c0} = 1,\quad (\mathrm{inv\_dx}\cdot\Delta x)_{c1} = 0.
     \]
6. **Doubling / inverse declarations**:
   - If \(b_d=1\): enforce \(\Delta x=0\) and \(\Delta y=0\) component-wise.
   - If \(b_i=1\): enforce \(\Delta x=0\) and \(y_Q+y_P=0\) component-wise.
7. **Slope equation**:
   - Add-branch: \(\Delta x\cdot\lambda = \Delta y\) in \(\mathbb{F}_{q^2}\) (2 component constraints).
   - Double-branch: \(2y_P\cdot\lambda = 3x_P^2\) in \(\mathbb{F}_{q^2}\) (2 component constraints), where multiplications use
     \((a_0+a_1u)(b_0+b_1u)=(a_0b_0-a_1b_1)+(a_0b_1+a_1b_0)u\).
8. **Result encoding and affine formulas**:
   - Inverse: \(b_i=1 \Rightarrow \mathrm{ind}_R=1\) (and infinity encoding forces all coords 0).
   - Non-inverse: \(\mathrm{ind}_R=0\) and the affine formulas
     \(x_R=\lambda^2-x_P-x_Q\), \(y_R=\lambda(x_P-x_R)-y_P\) hold component-wise.

The sumcheck batching/round structure matches G1Add (`num_vars = 0`, degree bound 6). Stage 2 caches the corresponding virtual openings for
wiring using `VirtualPolynomial::g2_add_*` helpers:

- `VirtualPolynomial::g2_add_xp_c0(i)`, `VirtualPolynomial::g2_add_xp_c1(i)`, `VirtualPolynomial::g2_add_yp_c0(i)`, `VirtualPolynomial::g2_add_yp_c1(i)`, `VirtualPolynomial::g2_add_p_indicator(i)`
- `VirtualPolynomial::g2_add_xq_c0(i)`, `VirtualPolynomial::g2_add_xq_c1(i)`, `VirtualPolynomial::g2_add_yq_c0(i)`, `VirtualPolynomial::g2_add_yq_c1(i)`, `VirtualPolynomial::g2_add_q_indicator(i)`
- `VirtualPolynomial::g2_add_xr_c0(i)`, `VirtualPolynomial::g2_add_xr_c1(i)`, `VirtualPolynomial::g2_add_yr_c0(i)`, `VirtualPolynomial::g2_add_yr_c1(i)`, `VirtualPolynomial::g2_add_r_indicator(i)`
- `VirtualPolynomial::g2_add_lambda_c0(i)`, `VirtualPolynomial::g2_add_lambda_c1(i)`, `VirtualPolynomial::g2_add_inv_delta_x_c0(i)`, `VirtualPolynomial::g2_add_inv_delta_x_c1(i)`
- `VirtualPolynomial::g2_add_is_double(i)`, `VirtualPolynomial::g2_add_is_inverse(i)`

---

#### Connection to the external pairing boundary

The Dory verifier’s final optimized check is a **3-way multi-pairing** (see the `dory` crate’s `DoryVerifierState::verify_final`).
We do not prove pairings inside this SNARK. Instead:
- the recursion witness includes all non-pairing operations that produce the three pairing input pairs, and
- the recursion SNARK exposes those three (G1,G2) pairs as public outputs,
so the outside verifier can perform the final pairing check.

---

## 3. Stage 2: Wiring / Boundary Constraints (Copy Constraints)

> **Implementation status**: AST-derived wiring/boundary constraints are **implemented** and enabled by default.
> They run as sumcheck instances appended at the end of the Stage 2 batched sumcheck (see `jolt-core/src/zkvm/recursion/{gt,g1,g2}/wiring.rs`
> and, in fully fused mode, `.../{gt,g1,g2}/fused_wiring.rs`).
> The design doc is `WIRING_SUMCHECK_PLAN.md` (repo root), plus this Section 3.

After Stage 2, we have many virtual polynomial claims $(v_0, v_1, \ldots, v_{n-1})$ at a shared point $r_x$.

Stage 2 enforces that all operation instances form **one coherent computation DAG** (copy constraints),
rather than a multiset of unrelated valid operations.

#### Topology source: Dory’s `AstGraph`

Dory can record the full verification computation as an AST/DAG (`AstGraph`) where each node:
- produces a typed `ValueId` (G1/G2/GT),
- records an `AstOp` (e.g. `G1Add`, `G2ScalarMul`, `GTMul`, `GTExp`), and
- lists its input `ValueId`s (the data-flow edges).

Wiring constraints are derived **only** from this graph structure: for every node input `ValueId`, we add a copy constraint
from the producer’s output value to the consumer’s input port. This avoids trusting prover-supplied wiring metadata.

### 3.2 What is being wired: typed values (ports)

We wire **typed values** (G1/G2/GT elements) between operation instances:
- `GTExp` output → `GTMul` input
- `GTMul` output → downstream `GTMul` input
- `G1ScalarMul` output → `G1Add` input
- `G2ScalarMul` output → `G2Add` input
- `G1Add`/`G2Add` outputs → downstream inputs, etc.

**Important soundness invariant (AST inputs):**
The wiring plan (`wiring_plan::derive_wiring_plan`) only creates copy-constraint edges from values that are **produced by proven ops**
(GTExp/GTMul/G1ScalarMul/G1Add/G2ScalarMul/G2Add). If an AST input value feeds directly into an op port without being produced by a proven op
or being represented as a boundary constant/public input for that op family, then the corresponding port could be left **unbound** by wiring.

In other words, for soundness we require:
- any GT value consumed by `GTMul` must be produced by `GTExp`/`GTMul` (or be handled as a boundary constant via the GT wiring plan),
- any G1/G2 value consumed by `G1Add`/`G2Add` must be produced by `G1ScalarMul`/`G1Add` or `G2ScalarMul`/`G2Add` respectively (or be a pairing-boundary constant),
- any “non-input base” used by `GTExp` / `G{1,2}ScalarMul` must be bound via base-binding edges (when the base itself is produced by a proven op).

If Dory’s AST semantics change to allow additional direct-input edges, `wiring_plan.rs` must be extended with new explicit boundary endpoints.

### 3.3 Port extraction for packed / step-indexed traces

Some operations are encoded as larger tables (e.g., packed GT exp, scalar-mul traces). In those cases, the *wired port*
is an **endpoint** (e.g., the final accumulator), not the entire trace.

We define endpoint ports using selector polynomials:
- **Packed GT exp**: $\rho(s,x)$ is an 11-var MLE with 7 step vars and 4 element vars (layout `index = x * 128 + s`).
  The output port is the last-step slice $\rho(s=\text{last}, x)$.
- **G1/G2 scalar mul**: trace columns are native 8-var MLEs over the step index.
  The output port is the last-step accumulator $A(\text{last})$ (including infinity indicator).

These endpoint ports can be expressed as *linear* constraints over the underlying MLEs using Eq selectors.
For example, to extract the last-step value of a step-indexed trace \(A\) (with 8 step vars \(s\)) at a random point \(\tau\) over any
additional “element vars” \(x\), one can enforce:
\[
\sum_{s,x} \mathrm{Eq}(\mathrm{last}, s)\cdot \mathrm{Eq}(\tau, x)\cdot \big(A(s,x) - B(x)\big)=0,
\]
which is equivalent to \(A(\mathrm{last},x)=B(x)\) for all \(x\).

**Suffix-alignment caveat**: Stage-2 sumcheck instances are batched with suffix alignment. Concretely, the Stage-2 challenge vector has
length equal to the maximum number of rounds among included instances (typically 11 due to packed GT gadgets). For native 8-var traces (G1/G2
scalar-mul and their shift checks), the effective opening point is the **suffix** of length 8 of the Stage-2 challenge vector.

For G2 ports, there is no single “element var” \(x\); instead, the \(\mathbb{F}_{q^2}\) coordinates are split into (c0,c1) polynomials.
Wiring a G2 element is therefore typically done by checking a random linear combination of its components (Fiat–Shamir batching).

### 3.4 Wiring check

For each type (G1/G2/GT), we check a random linear combination of all copy constraints at the shared point $r_x$.
At a high level, for edges \(e\) we check:
\[
0 \stackrel{!}{=} \sum_{e} \lambda_e \cdot (\text{PortOut}_{\text{src}(e)}(r_x) - \text{PortIn}_{\text{dst}(e)}(r_x)),
\]
where \(\lambda_e\) are transcript challenges (Fiat–Shamir) to prevent cancellation.

### 3.5 Fully fused wiring (GT/G1/G2)

When fully fused mode is enabled for a family, Stage 2 uses fused wiring backends:

- GT: `gt/fused_wiring.rs`
- G1: `g1/fused_wiring.rs`
- G2: `g2/fused_wiring.rs`

These backends are designed to:

- run over a shared Stage-2 challenge point that includes an additional constraint-index segment `c_common`,
- implement split-\(k\) dummy handling via the shared convention in Section 1.4.1, and
- **consume cached fused openings** (verifier-side) instead of requesting per-edge/per-instance PCS openings.

In fully fused mode:

- `G1ScalarMul` caches fused port openings under `SumcheckId::G1ScalarMul` (see `g1/fused_scalar_multiplication.rs`).
- `G2ScalarMul` caches fused port openings under `SumcheckId::G2ScalarMul` (see `g2/fused_scalar_multiplication.rs`).
- `G1Add` caches fused port/aux openings under `SumcheckId::G1Add` (see `g1/fused_addition.rs`).
- `G2Add` caches fused port/aux openings under `SumcheckId::G2Add` (see `g2/fused_addition.rs`).
- Fused wiring instances cache **no openings**; they only add a Stage-2 sumcheck equation that binds those openings to the verifier-derived wiring plan.

### 3.6 Soundness audit checklist (wiring + fused split-\(k\))

This checklist is intended for “red-team” review of the fused protocols.

- **Wiring-plan edge coverage** (`wiring_plan.rs`):
  - Ensure every AST value that flows into an op port is either:
    - produced by a proven op (and therefore gets a wiring edge), or
    - explicitly represented as a boundary constant / public input for that family.
  - Pay special attention to *direct AST inputs* feeding `GTMul` / `G1Add` / `G2Add` ports. If they exist, wiring must be extended.

- **Split-\(k\) correctness** (dummy-bit convention):
  - Dummy bits must be **low bits** of `c_common`, and family bits must be the suffix.
  - Any selector replicated across dummy bits must be paired with \(\beta(\text{dummy}) = 2^{-\text{dummy}}\) normalization.
  - Verify both endpoints of each edge use the correct family’s `k_*` and the correct \(\beta\).

- **Opening-point normalization** (fused scalar-mul):
  - Fused scalar-mul openings must be cached at `(step, c_tail)` (dropping dummy low bits), and any consumer (shift/wiring)
    must use the same normalized point.

- **Ordering / cache dependencies** (Stage 2 instance list):
  - `FusedGtExpStage2Openings*` must run before `FusedGtShift*` and fused GT wiring.
  - Fused scalar-mul must run before fused shift (G1/G2).
  - Wiring instances must run last.

- **Boundary bindings**:
  - Combine-commitments root must be bound to `joint_commitment`.
  - Pairing boundary \(p1,p2,p3\) (G1+G2) and RHS (GT) must be bound by wiring edges derived from the `MultiPairing` node.

---

## 4. Stage 3: Prefix Packing Reduction

Stage 3 replaces the previous “direct-eval → jagged transform” pipeline.
Its purpose is to **reduce many Stage-2 virtual openings** (native-size witness polynomials evaluated at the shared Stage-2 point)
to a **single PCS opening claim** of one packed multilinear polynomial.

### 4.1 Public packing layout (canonical)

Both prover and verifier deterministically derive a packing layout from the **public constraint list** (`constraint_types`):

- Each constraint family contributes a fixed set of witness polynomials (indexed by `PolyType`) with known native arities (e.g. 4-var, 8-var, 11-var).
- We order these polynomials canonically and assign them disjoint prefix-defined subcubes of \(\{0,1\}^{n_{\text{dense}}}\).
- Unused regions of the ambient \(\{0,1\}^{n_{\text{dense}}}\) hypercube are treated as zero.

#### Canonical ordering specification

The polynomials are sorted using the following **lexicographic sort key** (applied to each committed polynomial):

1. **`num_vars` descending**: Larger polynomials (more variables) come first.
2. **`PolyType` ascending**: Within same size, order by `PolyType` enum discriminant.
3. **`constraint_idx` ascending**: Within same size and type, order by constraint index in the public constraint list.

This ordering guarantees:
- **Alignment**: Power-of-two alignment is maintained (larger blocks placed first ensures subsequent blocks align).
- **Determinism**: The layout is identical for prover and verifier, computed from public data only.
- **Stability**: The ordering is independent of prover choices and stable across implementations.

**Compatibility note**: This ordering is part of the recursion proof semantics and therefore part of the
serialized proof/encoding. Any future modifications must be versioned (and artifacts regenerated) to avoid
proof incompatibility.

This is implemented by `PrefixPackingLayout::from_constraint_types` in `jolt-core/src/zkvm/recursion/prefix_packing.rs`.
When any end-to-end fused mode is enabled (GT and/or G1 and/or G2), the code uses
`PrefixPackingLayout::from_constraint_types_fused(...)` instead.

### 4.2 Stage 3 protocol

**Inputs**:
- Stage-2 batched point `r_stage2` (in sumcheck round order; shorter Stage-2 instances are suffix-aligned in the batched sumcheck)
- Stage-2 virtual opening claims \(v_e\) cached in the opening accumulator (for legacy families these are per-instance claims at `r_x`;
  for fused families these are fused claims at a normalized opening point, e.g. `(step, c_tail)` for scalar-mul fusion)

**Protocol**:
1. **Sample**: fresh packing challenges `r_pack` from the transcript (Fiat–Shamir).
2. **Form**: packed opening point `r_full = [r_x || r_pack]` (implementation maps `r_x` into the packed low-variable order).
3. **Compute**: packed evaluation
   \[
   \mathrm{packed\_eval} \;=\; \sum_{e \in \text{layout}} w_e(r_{\text{pack}})\cdot v_e,
   \]
   where \(w_e\) is the multilinear weight for the prefix-coded subcube selector of entry \(e\).
4. **Prover sends** `stage3_packed_eval = packed_eval` (one field element) and appends it to the transcript.
5. **Both sides register** a PCS opening claim for the committed packed polynomial at point `r_full`.

**Output**: a single dense opening claim \(\mathrm{PackedPoly}(r_{\text{full}})=\mathrm{stage3\_packed\_eval}\).

### 4.3 Implementation note: suffix alignment and variable order

Stage-2 sumchecks bind variables in `BindingOrder::LowToHigh` (LSB-first) and are **suffix-aligned** in the batched sumcheck.
Stage 3 therefore interprets `r_x` as a suffix and maps it into the low variables of the packed opening point in the same order expected by
the prefix packing layout.

**Bit-reversal for claim consistency**: Before embedding `r_x` as the low bits of the packed opening point, we **reverse** the
vector (so that the Stage-2 suffix becomes a prefix in packed coordinates). Correspondingly, each native polynomial's evaluation
table is bit-reversed when building the packed dense polynomial. This ensures that
\(\mathrm{packed\_eval} = \sum_e w_e(r_{\mathrm{pack}}) \cdot f_e(r_x)\) correctly reduces to the committed table values.

For the precise mapping, see:
- `jolt-core/src/zkvm/recursion/prefix_packing.rs` (`build_prefix_packed_evals`, `packed_eval_from_claims`)
- `RecursionProver::prove_stage3_prefix_packing` / `RecursionVerifier::verify_stage3_prefix_packing`

---

## 5. Stage 4: Jagged Transform Sum-Check

> **Removed**: Stages 4 and 5 were a previous “jagged polynomial commitment” pipeline.
> The recursion implementation now uses **prefix packing** (Stage 3) and does not execute a jagged transform / assist protocol.
> If we ever need to revisit the jagged approach, we can recover it from git history.

## 6. PCS Opening Proof (Hyrax over Grumpkin)

After Stage 3 (prefix packing reduction), we have:
- a single packed polynomial commitment target (the prefix-packed dense polynomial), and
- an accumulated set of opening claims stored in the opening accumulator.

This final PCS step proves those openings using Hyrax over Grumpkin.

### 6.1 Why Hyrax

- **No pairing required**: Works over any curve with efficient MSM
- **Grumpkin-native**: Scalar field = $\mathbb{F}_q$, matching our constraint field
- **Square matrix efficiency**: Optimal for virtualized polynomials

### 6.2 Commitment Structure

For polynomial $q$ with $2^n$ evaluations:
1. Reshape as $2^{n_r} \times 2^{n_c}$ matrix
2. Commit to each row via Pedersen over Grumpkin
3. Commitment: $C = \{C_0, \ldots, C_{2^{n_r}-1}\}$

### 6.3 Opening Protocol

**Input**: Commitment $C$, point $r_{\text{full}}$, claimed value $\mathrm{stage3\_packed\_eval}$

**Protocol**:
1. Decompose point into row/column components
2. Sum-check proves tensor product structure
3. Column opening proves consistency
4. Verifier checks against commitment

### 6.4 Final Verification

The verifier accepts iff all checks pass:
- Stage 1 and Stage 2 sumcheck verifications (including shift checks and, when implemented, wiring/boundary checks)
- Stage 3 (prefix packing reduction) check
- PCS opening proof verification (Hyrax over Grumpkin)

---

## 7. Parameters & Cost Analysis

This section provides analytical formulas for proof sizes, constraint counts, and computational costs.

### 7.1 Sumcheck Degrees and Rounds

| Stage | Protocol | Degree | Rounds | Elements/Round |
|-------|----------|--------|--------|----------------|
| 1 | GT Exponentiation (unpacked) | 4 | 4 | 5 |
| 1 | GT Exponentiation (packed) | 7 | 11 | 8 |
| 2 | GT Multiplication | 3 | 4 | 4 |
| 2 | G1 Scalar Multiplication | 6 | 8 | 7 |
| 2 | Shift G1/G2 Scalar Mul | 3 | 8 | 4 |
| 2 | Wiring / boundary constraints | - | - | - |
| 3 | Prefix Packing Reduction | - | 0 | 1 |

**Note**: Stage-2 instances are batched in one `BatchedSumcheck`. Instances with fewer rounds are suffix-aligned via `round_offset`.

Where:
- $\ell = \lceil \log_2 n \rceil$ for $n$-bit scalar
- \(n_{\text{dense}} = \lceil \log_2(\text{dense\_size}) \rceil\) is the number of variables of the packed polynomial.

### 7.2 Constraint Counts

**Per-operation constraints** (high level):

- **GT Exp (unpacked, \(t\)-bit)**: \(t\) constraints, 4 polynomial types per bit (base, rho-prev, rho-curr, quotient).
- **GT Exp (packed, base-4)**: 1 packed constraint covering up to 128 base-4 steps. The packed witness tables include
  \((\rho,\rho_{\text{next}},Q)\), and Stage 2 runs `GtShift` to enforce the global shift relation \(\rho_{\text{next}}(s,x)=\rho(s+1,x)\).
- **GT Mul**: 1 constraint, 4 polynomial types (lhs, rhs, result, quotient).
- **G1 Scalar Mul (256-bit)**: 1 constraint over an 8-var step domain, with accumulator/double/next coordinates plus indicators.
- **G2 Scalar Mul (256-bit)**: same shape as G1 but with Fq2 split into (c0,c1) components and additional constraints.
- **G1Add / G2Add (new)**: 1 constraint per add node (with the exact polynomial interface defined by the add sumcheck implementation).

**Implementation note**: The exact set of “poly types” that become matrix rows is defined by the recursion constraint system
(`jolt-core/src/zkvm/recursion/constraints/system.rs`) and is the authoritative source of truth.

### 7.3 Matrix Dimensions

**Row count**:
$$\text{num\_rows} = \text{NUM\_POLY\_TYPES} \times c_{\text{pad}}$$

where $c_{\text{pad}} = 2^{\lceil \log_2 c \rceil}$ and $c$ = total constraints.

**Note**: `NUM_POLY_TYPES` is an implementation constant defined by the recursion constraint system and grows as we add op types
(e.g., G2 scalar mul, G1/G2 add) and/or change packing layouts.

**Column count**:
- $2^4 = 16$ for GT operations (4-variable MLEs)
- $2^8 = 256$ for G1 operations (8-variable MLEs)
- Sparse matrix uses $2^8 = 256$ (zero-padded)

**Example** ($c = 10$ constraints):

| Parameter | Formula | Value |
|-----------|---------|-------|
| $c_{\text{pad}}$ | $2^{\lceil \log_2 10 \rceil}$ | 16 |
| num\_rows | $\text{NUM\_POLY\_TYPES} \times 16$ | (depends on implementation) |
| num\_s\_vars | $\lceil \log_2(\text{num\_rows}) \rceil$ | (depends on implementation) |

### 7.4 Dense Size Computation

Prefix packing builds a single packed polynomial by concatenating native-size witness polynomials without padding:

$$\text{dense\_size} = \sum_{\text{poly } p} 2^{\text{num\_vars}(p)}$$

where:
- GT polynomials: $\text{num\_vars} = 4 \Rightarrow 16$ evaluations
- G1 polynomials: $\text{num\_vars} = 8 \Rightarrow 256$ evaluations

**Example** (3 GT exp + 2 GT mul + 1 G1 scalar mul):

| Source | Polys | Evals/Poly | Total |
|--------|-------|------------|-------|
| GT Exp (3 constraints × 4 types) | 12 | 16 | 192 |
| GT Mul (2 constraints × 4 types) | 8 | 16 | 128 |
| G1 Scalar Mul (1 constraint × 7 types) | 7 | 256 | 1,792 |
| **Total** | 27 | — | **2,112** |

$\Rightarrow n_{\text{dense}} = \lceil \log_2 2112 \rceil = 12$ packed variables.
Stage 3 itself is sumcheck-free (it contributes one field element `stage3_packed_eval` plus a PCS opening claim).

### 7.5 Proof Size Formulas

**Stage 1** (sumcheck messages):
$$|P_1| = \sum_{\text{type } t} (\text{degree}_t + 1) \times \text{rounds}_t$$

| Type | Degree | Rounds | Elements |
|------|--------|--------|----------|
| GT Exp (unpacked) | 4 | 4 | 20 |
| GT Exp (packed) | 4 | 12 | 60 |
| GT Mul | 3 | 4 | 16 |
| G1 Scalar Mul | 6 | $\ell$ | $7\ell$ |

**Stage 2** (batched sumcheck):
The Stage-2 proof size is \(O(\text{max\_num\_rounds} \cdot \text{max\_degree})\) field elements (one batched sumcheck proof).

**Stage 3** (prefix packing reduction):
$$|P_3| = 1 \text{ element} \quad (\mathrm{stage3\_packed\_eval})$$

**Legacy (removed)**: Stage 4/5 jagged transform + assist are no longer part of the proof.

**PCS opening proof** (Hyrax over Grumpkin):
$$|P_{\text{pcs}}| = O(\sqrt{\text{dense\_size}}) \text{ group elements}$$

**Total proof size** (field elements, excluding PCS):
$$|P| = |P_1| + |P_2| + |P_3| + \text{virtual claims}$$

### 7.6 Concrete Example: Single 256-bit GT Exponentiation (Unpacked vs Packed)

This example isolates just the Stage 1 representation cost for a single 256-bit exponentiation.

**Unpacked**:
- Constraints: \(t=256\) (one per bit)
- Virtual claims: \(4t = 1024\) (base, rho-prev, rho-curr, quotient per bit)
- Sumcheck rounds: 4 (over the 4 element variables)

**Packed (base-4)**:
- Constraints: 1 (covers up to 128 base-4 steps)
- Virtual claims: 3 (\(\rho,\rho_{\text{next}},Q\)) at \((r_s^*,r_x^*)\)
- Sumcheck rounds: 11 (7 step vars + 4 element vars)

**Note**: End-to-end Stage 2/3/4 costs depend on the full recursion constraint system (all op types in scope, padding rules, and
`NUM_POLY_TYPES`), so they are intentionally not hard-coded in this single-op example.

### 7.7 Prover Complexity

| Stage | Operation | Complexity |
|-------|-----------|------------|
| Stage 1 | Sumcheck per round | $O(2^{\text{vars}} \cdot \text{degree})$ |
| Stage 2 | Batched sumchecks | $O(\#\text{instances} \cdot 2^{11} \cdot \text{degree})$ |
| Stage 3 | Prefix packing reduction | $O(\#\text{packed polys} \cdot n_{\text{dense}})$ field ops |
| PCS | Hyrax commitment/opening | $O(\text{dense\_size})$ MSMs |

**Dominant cost**: Stage 1 sumcheck computation scales with constraint count.

**Parallelization**: Stage 1's three constraint types run in parallel.

### 7.8 Verifier Complexity

| Stage | Operation | Complexity |
|-------|-----------|------------|
| Stage 1 | Sumcheck verification | $O(\text{rounds} \cdot \text{degree})$ |
| Stage 2 | Batched sumcheck verification | $O(\text{rounds} \cdot \text{degree})$ |
| Stage 3 | Prefix packing reduction | $O(\#\text{packed polys} \cdot n_{\text{dense}})$ field ops |
| PCS | Hyrax verification | $O(\sqrt{\text{dense\_size}})$ |

**Key efficiency**: prefix packing avoids Stage 4/5 entirely; the verifier computes `stage3_packed_eval` in time linear in the number of packed polynomials and then checks one PCS opening.

### 7.9 Comparison: Sparse vs Dense

> **Legacy note**: The discussion below compares the removed jagged pipeline to a sparse, padded matrix representation.
> With prefix packing, we commit directly to the packed polynomial of size `dense_size` (no padding) and do not run Stage 4/5.

Without jagged transform (sparse):
- Matrix size: $15 \cdot c_{\text{pad}} \times 256$
- For 256 GT exp constraints: $3,840 \times 256 = 983,040$ entries

With jagged transform (dense):
- Dense size: $1,024 \times 16 = 16,384$ entries
- **Compression ratio**: $60\times$

This compression directly reduces:
- PCS commitment size
- Stage 4 prover work
- Hyrax opening proof size

### 7.10 Scaling Summary

| Metric | Formula | 256-bit GT Exp |
|--------|---------|----------------|
| Constraints | $t$ (bit-length) | 256 |
| Stage 1 rounds | 11 (packed GT) | 11 |
| Stage 3 rounds | $\lceil \log_2(15c) \rceil$ | 12 |
| Stage 4 rounds | $\lceil \log_2(4tc \cdot 16) \rceil$ | 14 |
| Proof elements | $O(c + \log c)$ | ~1,100 |
| Prover time | $O(c \cdot 2^{11})$ | ~4,000 ops |
| Verifier time | $O(c + \text{polys} \cdot \text{bits})$ | ~15,000 ops |

The protocol achieves **logarithmic scaling** in proof size relative to constraint count, with linear prover work and near-linear verifier work.

### 7.11 Unified Polynomial Cost Analysis

The packed GT exponentiation optimization dramatically reduces system costs:

**Memory Requirements**:

| Approach | Polynomials | Memory (256-bit exp) | Formula |
|----------|-------------|---------------------|---------|
| Unpacked | $4t$ | ~32 MB | $4t \times 2^4 \times 32$ bytes |
| Packed | 3 | ~0.2 MB | $3 \times 2^{11} \times 32$ bytes |

**Stage 3 Virtual Claims**:

| Approach | Virtual Claims | Verifier Work |
|----------|---------------|---------------|
| Unpacked | 1,024 | $O(1024)$ field ops |
| Packed | 3 | $O(3)$ field ops |

**Impact on Later Stages**:
- Stage 3 processes fewer polynomials (3 vs 1,024)
- Stage 5 benefits from reduced $K$ in batch verification
- Hyrax commitment is more efficient with fewer polynomials

**Trade-offs**:
- Packed approach uses 11-round sumcheck vs 4-round
- Slightly larger Stage 1 proof (60 vs 20 elements)
- Prover computes over larger domain ($2^{11}$ vs $2^4$)
- Net benefit: ~9.5× smaller total proof, ~200× fewer virtual claims

---

## 8. Implementation

This section describes the code architecture and data flow for the recursion implementation.

### 8.1 Module Structure

```
jolt-core/src/zkvm/recursion/
├── mod.rs
├── spec.md
├── witness.rs
├── prover.rs                  # orchestrates stages
├── verifier.rs
├── prefix_packing.rs          # Stage 3 (prefix packing reduction)
├── virtualization.rs          # helper utilities for virtual claims
├── constraints/               # constraint-system + matrix layout
│   ├── config.rs
│   ├── sumcheck.rs
│   └── system.rs
├── gt/                        # GT gadgets + auxiliary subprotocols
│   ├── exponentiation.rs      # Stage 1 (`GtExp`)
│   ├── shift.rs               # Stage 2 (`GtShift`)
│   ├── claim_reduction.rs     # Stage 2 (`GtExpClaimReduction`)
│   └── multiplication.rs      # Stage 2 (`GtMul`)
├── g1/
│   ├── scalar_multiplication.rs
│   ├── addition.rs
│   └── shift.rs               # Stage 2 (`ShiftG1ScalarMul` / `ShiftG2ScalarMul`)
├── g2/
│   ├── scalar_multiplication.rs
│   └── addition.rs
├── pairing/
│   ├── multi_miller_loop.rs   # Stage 2 (`MultiMillerLoop`)
│   └── shift.rs               # Stage 2 (`ShiftMultiMillerLoop`)
```

**Note on stage naming**: The “Stage 1..3” names refer to protocol phases (see Section 1.4 and the prover/verifier in `prover.rs` / `verifier.rs`).
The code is organized by gadget family (`gt/`, `g1/`, `g2/`, `pairing/`) plus the shared back-end (`constraints/`, `prefix_packing.rs`, `virtualization.rs`).

### 8.2 The Offloading Pattern

The revised design treats the **Dory proof** as the ground truth and does **not** rely on a “HintMap” that supplies intermediate
group elements to the verifier.

The offloading boundary is:

- **Inside the recursion SNARK**: all non-pairing group/GT computation implied by Dory verification (scalar mul, add, GT exp/mul, wiring).
- **Outside the recursion SNARK**: transcript hashing / Fiat–Shamir challenge derivation, scalar-field arithmetic (inverses/products), and the
  **final 3-way multi-pairing check**.
  The verifier also derives the recursion verifier input using hint-based plan derivation (no expensive group operations); with wiring enabled,
  Stage 2 binds the pairing boundary and non-input bases/points to the verifier-derived AST.

At a high level:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                PROVER SIDE                                 │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Dory proof + setup + transcript  ──→  run Dory verify in witness-gen mode │
│                                       (with AST tracing enabled)           │
│                                  ──→  (WitnessCollection, AstGraph)        │
│                                                                            │
│  (WitnessCollection, AstGraph) ──→  RecursionProver ──→ RecursionProof      │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                               VERIFIER SIDE                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  derive transcript challenges / scalars outside SNARK                       │
│  RecursionVerifier verifies RecursionProof                                  │
│    - obtains three pairing input pairs (p1,p2,p3) (and optionally rhs)      │
│  outside verifier computes multi_pair(p1,p2,p3) and checks == rhs           │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**Implementation note**: Dory already supports collecting an AST during verification via
`TraceContext::for_witness_gen_with_ast()` and extracting it via `finalize_with_ast()`.

### 8.3 Witness Types

Each expensive operation has a corresponding witness structure:

**GT Exponentiation** (`GTExpWitness`):
```rust
struct GTExpWitness {
    base: Fq12,                    // Base element a
    exponent: Fr,                  // Scalar k
    result: Fq12,                  // Result a^k
    rho_mles: Vec<Vec<Fq>>,        // Intermediate ρ values as MLEs
    quotient_mles: Vec<Vec<Fq>>,   // Quotient Q_i for each step
    bits: Vec<bool>,               // Binary decomposition of k
}
```

**GT Multiplication** (`GTMulWitness`):
```rust
struct GTMulWitness {
    lhs: Fq12,                     // Left operand a
    rhs: Fq12,                     // Right operand b
    result: Fq12,                  // Result c = a·b
    quotient_mle: Vec<Fq>,         // Quotient polynomial
}
```

**G1 Scalar Multiplication** (`G1ScalarMulWitness`):
```rust
struct G1ScalarMulWitness {
    constraint_index: usize,
    base_point: (Fq, Fq),          // P = (x_P, y_P) (public/wired input)
    // 11-var MLE eval vectors (size 2^11) for the 256-step trace (8 step bits, padded to 11 vars)
    x_a: Vec<Fq>,                  // x_A(s)
    y_a: Vec<Fq>,                  // y_A(s)
    x_t: Vec<Fq>,                  // x_T(s) = x([2]A_s)
    y_t: Vec<Fq>,                  // y_T(s) = y([2]A_s)
    x_a_next: Vec<Fq>,             // x_A(s+1) (shifted)
    y_a_next: Vec<Fq>,             // y_A(s+1) (shifted)
    a_indicator: Vec<Fq>,          // ind_A(s)
    t_indicator: Vec<Fq>,          // ind_T(s)
}
```

The scalar \(k\) is treated as a **public input** (the outside verifier derives its bits from the transcript), so the recursion
SNARK does not require a committed bit polynomial.

**G2 Scalar Multiplication** (`G2ScalarMulWitness`):

```rust
struct G2ScalarMulWitness {
    constraint_index: usize,
    base_point: (Fq2, Fq2),        // Q = (x_Q, y_Q) (public/wired input)
    // Fq2 coordinates are split into c0/c1 components in Fq
    x_a_c0: Vec<Fq>, x_a_c1: Vec<Fq>,
    y_a_c0: Vec<Fq>, y_a_c1: Vec<Fq>,
    x_t_c0: Vec<Fq>, x_t_c1: Vec<Fq>,
    y_t_c0: Vec<Fq>, y_t_c1: Vec<Fq>,
    x_a_next_c0: Vec<Fq>, x_a_next_c1: Vec<Fq>,
    y_a_next_c0: Vec<Fq>, y_a_next_c1: Vec<Fq>,
    a_indicator: Vec<Fq>,
    t_indicator: Vec<Fq>,
}
```

**G1 Addition** (`G1AddWitness`) and **G2 Addition** (`G2AddWitness`) (new):

These witnesses provide the input points and output point for each explicit `G1Add`/`G2Add` node in Dory’s AST, including
infinity encoding. A corresponding Stage 1 sumcheck enforces the affine group law (with correct infinity handling).

**Packed GT Exponentiation** (Used when optimization enabled):

The packed representation combines all steps into a single table over step bits and element bits.
The implementation uses **base-4 digits** (two bits per step), so there are at most 128 steps and the packed tables
use **11 variables** total (7 step vars + 4 element vars).

```rust
struct GtExpWitness {
    // 11-var packed tables (size 2^11 = 2048)
    rho_packed: Vec<Fq>,       // ρ(s,x)
    rho_next_packed: Vec<Fq>,  // ρ(s+1,x)
    quotient_packed: Vec<Fq>,  // Q(s,x)

    // Public-input-derived tables (replicated across the other dimension)
    digit_lo_packed: Vec<Fq>,  // digit_lo(s)
    digit_hi_packed: Vec<Fq>,  // digit_hi(s)
    base_packed: Vec<Fq>,      // base(x)
    base2_packed: Vec<Fq>,     // base^2(x)
    base3_packed: Vec<Fq>,     // base^3(x)
}
```

Layout: For 11-variable MLEs with `s ∈ {0,1}^7` and `x ∈ {0,1}^4`:
- Index formula: `index = x * 128 + s` (s in low bits)
- `rho_packed[x * 128 + s] = ρ_s[x]`
- Each packed table has 2^11 = 2,048 evaluations

### 8.4 Constraint System Construction

**Current code path**: the streaming recursion constraint system is planned from the **witness stores**, not from the AST topology.

The planner is `witness_generation::plan_constraint_system` (`jolt-core/src/zkvm/recursion/witness_generation.rs`), which takes:
- a `WitnessCollection<JoltWitness>` (from `PCS::witness_gen_with_ast`), and
- an optional `GTCombineWitness` (from `PCS::generate_combine_witness`),
and produces a `ConstraintSystem` containing:
- the public `constraint_types` list (this is the verifier “program” for Stage 2/3), and
- per-family native witness stores (GT mul 4-var, scalar-mul traces 8-var, add ports 0-var, etc.).

**Determinism rule (important for prover/verifier agreement)**:
- Each witness map is sorted by `OpId` before being appended to the plan (see `witness_generation.rs`, GT exp/GT mul/G1/G2 sections).
- Combine witnesses are appended after the direct witnesses (exp first, then muls in layer order).

**Role of the AST today**:
- The AST is captured on the prover side and re-derived on the verifier side for binding and for wiring/copy constraints (via a `WiringPlan`).
- The AST is **not** used to define the Stage-2 `constraint_types` ordering; that ordering is derived from witness collection ordering and must
  match between prover and verifier.

### 8.5 Prover Flow

**Current code path**: `RecursionProver::prove` (`jolt-core/src/zkvm/recursion/prover.rs`) is the end-to-end entrypoint. It is a 4-phase pipeline:

1. **Witness generation (Stage 8/9 boundary)**: `RecursionProver::witness_generation`
   - compute Stage-8 combine witness (`PCS::generate_combine_witness`)
   - run Stage-9 witness generation with AST capture (`PCS::witness_gen_with_ast`)
   - derive a `PairingBoundary` from the AST
   - build the recursion `ConstraintSystem` via `witness_generation::plan_constraint_system`

2. **Commit (Hyrax)**: `RecursionProver::poly_commit`
   - emits the prefix-packed dense polynomial via `witness_generation::emit_dense`
   - derives `RecursionConstraintMetadata` (notably `constraint_types`, `dense_num_vars`, and public inputs)
   - commits via Hyrax and appends the commitment to the transcript

3. **Sumchecks**: `RecursionProver::prove_sumchecks`
   - Stage 1: packed GT exp (`GtExp`) only
   - Stage 2: batched constraint sumchecks (shift + claim reduction + GT mul + G1/G2 scalar mul + G1/G2 add + wiring/boundary constraints)
   - Stage 3: prefix packing reduction (no sumcheck; registers a single committed opening claim)

4. **PCS opening**: `RecursionProver::poly_opening`
   - proves the accumulated openings against the Hyrax commitment

### 8.6 Verifier Flow

**Current code path**: `RecursionVerifier::verify` (`jolt-core/src/zkvm/recursion/verifier.rs`) mirrors the prover:

**Outer wrapper**: `verify_recursion` (`jolt-core/src/zkvm/recursion/api.rs`) replays base stages 1–7 to reconstruct transcript state, builds a
symbolic AST, derives a `RecursionVerifierInput` via hint-based plan derivation, verifies the recursion SNARK via `RecursionVerifier::verify`,
and finally performs the external 3-way pairing check against `recursion.pairing_boundary.rhs`.

- Bind the dense commitment into the transcript (must match prover ordering).
- Initialize the opening accumulator from `proof.opening_claims`.
- Verify:
  - Stage 1 packed GT exp sumcheck,
  - Stage 2 batched constraint sumchecks (including wiring/boundary constraints),
  - Stage 3 prefix-packing reduction (recompute `stage3_packed_eval`, then register the committed opening),
  - Hyrax opening proof verification.

### 8.7 Opening Accumulator

The `OpeningAccumulator` tracks virtual polynomial claims across stages:

```rust
// Example: a sumcheck instance caches virtual openings under a SumcheckId.
append_virtual_claims(
    accumulator,
    transcript,
    SumcheckId::GtExpClaimReduction,
    &opening_point,
    &virtual_claims![VirtualPolynomial::gt_exp_rho(i) => rho_eval],
);

// Stage 3 registers the single committed opening on the prefix-packed dense polynomial.
accumulator.append_dense(
    transcript,
    CommittedPolynomial::DoryDenseMatrix,
    SumcheckId::RecursionPacked,
    opening_point, // big-endian point (matches PCS verifier format)
    packed_eval,
);
```

### 8.8 Prefix Packing Layout

Stage 3 uses a deterministic **prefix packing** layout to place each committed witness polynomial into a disjoint prefix-defined subcube of one packed dense multilinear.

The canonical ordering, offsets, and the packed-variable mapping are implemented by `PrefixPackingLayout` in `jolt-core/src/zkvm/recursion/prefix_packing.rs`.

### 8.10 Polynomial Type Enumeration

The recursion matrix row layout is an encoding detail of the constraint system and evolves as we add new op types (e.g., G1/G2 add)
and refine packed protocols (e.g., packed GT exp). The authoritative definition is the `PolyType` enum in:

- `jolt-core/src/zkvm/recursion/constraints/system.rs`

### 8.11 Dory Integration: Witness + AST Extraction (No HintMap)

We run Stage 8/9 recursion witness generation (prover-side) in a way that records:

- a **WitnessCollection**: per-op witness traces for Stage 1 sumchecks, and
- an **AstGraph**: the full computation DAG used to derive wiring/copy constraints deterministically.

In the current codebase, this is done via the PCS recursion extension APIs in `RecursionProver::witness_generation` (`jolt-core/src/zkvm/recursion/prover.rs`):
- `PCS::witness_gen_with_ast(...) -> (WitnessCollection, AstGraph)`
- `PCS::generate_combine_witness(...) -> (GTCombineWitness, CombineHint)`

```rust
let (witness_collection, ast) = PCS::witness_gen_with_ast(
    stage8_opening_proof,
    verifier_setup,
    &mut witness_gen_transcript,
    &opening_point,
    &joint_claim,
    &joint_commitment,
)?;
let prover = RecursionProver::new_from_witnesses(&witness_collection, Some(combine_witness))?;
```

**Verifier-side topology**: the recursion verifier must not trust a prover-supplied AST. In the current verifier, Stage 8 recursion verification
already reconstructs a symbolic AST deterministically from public inputs (`jolt-core/src/zkvm/verifier.rs::verify_stage8_with_recursion`).
Wiring/copy constraints (when implemented) should use that verifier-derived AST.

### 8.12 GPU Considerations

The recursion prover has a clear separation between orchestration logic (Rust) and compute-intensive kernels (GPU).

#### GPU-Accelerated Components

| Component | Operation | Why GPU |
|-----------|-----------|---------|
| Stage 1 Sumcheck | Polynomial evaluations over hypercube | Parallel evaluation of $2^n$ points |
| Stage 2 Sumcheck | Batched constraint sumchecks | Parallel evaluation of $2^n$ points across many instances |
| Stage 3 Prefix Packing | Packing reduction | Efficient linear reduction over claims |
| Hyrax Commit | Multi-scalar multiplication (MSM) | $O(\sqrt{N})$ group operations |
| Hyrax VMP | Vector-matrix product | Parallelizable linear algebra |

#### Rust-Side Orchestration

The following remain on the CPU/Rust side:

- **Witness generation**: `TraceContext` collection during Dory verify
- **Transcript management**: Fiat-Shamir challenge generation
- **Prefix packing layout**: derive a canonical packing layout from public `constraint_types`
- **AST extraction**: `AstGraph` extraction and (verifier-side) deterministic reconstruction for wiring
- **Proof assembly**: Collecting stage outputs into `RecursionProof` (including the external pairing boundary outputs)

#### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RUST (CPU)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  Stage 8/9 recursion witness-gen ──→ WitnessCollection (+ AstGraph)      │
│  WitnessCollection (+ combine witness) ──→ ConstraintSystem              │
│                                              │                          │
│  Transcript ←─────────────────────────────── │ ←── challenges           │
│       │                                      │                          │
│       ▼                                      ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         GPU KERNELS                              │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  Stage 1: sumcheck_prove(polys, eq_evals) → round_polys         │   │
│  │  Stage 2: batched_sumcheck_prove(instances) → round_polys       │   │
│  │  Stage 3: prefix_pack_reduce(claims, layout) → packed_eval      │   │
│  │  Hyrax:   msm(scalars, bases) → commitments                     │   │
│  │           vmp(matrix, vector) → result                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  RecursionProof { stage1, stage2, stage3_packed_eval, opening }        │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Kernel Interface

Each sumcheck stage exposes a similar GPU interface:

```rust
trait SumcheckGpuKernel {
    /// Compute all round polynomials for one sumcheck
    fn prove_rounds(
        &self,
        evaluations: &GpuBuffer<F>,      // Polynomial evals on hypercube
        eq_evals: &GpuBuffer<F>,          // eq(r, x) precomputed
        challenges: &mut impl TranscriptReceiver,
    ) -> Vec<RoundPolynomial>;
}
```

Hyrax operations:

```rust
trait HyraxGpuKernel {
    /// Multi-scalar multiplication for row commitments
    fn msm(&self, scalars: &GpuBuffer<F>, bases: &GpuBuffer<G>) -> Vec<G>;

    /// Vector-matrix product for opening
    fn vmp(&self, matrix: &GpuBuffer<F>, vector: &[F]) -> Vec<F>;
}
```

#### Memory Considerations

| Stage | GPU Memory | Notes |
|-------|------------|-------|
| Stage 1 | $O(c \cdot 2^4)$ | Small per-constraint polynomials |
| Stage 2 | $O(2^s)$ | Full virtualized matrix |
| Stage 3 | $O(\text{dense\_size})$ | Compressed representation |
| Hyrax | $O(\sqrt{N})$ bases + scalars | MSM working set |

Prefix packing (Stage 3) reduces GPU memory pressure by avoiding padded sparse representations.

### 8.13 Environment Flags and Feature Gates

The recursion implementation includes several feature toggles (primarily for benchmarking/debugging):

- **Important (guest soundness)**: for end-to-end fused modes (GT/G1/G2), the verifier must not rely on environment variables. These booleans
  are carried in `RecursionVerifierInput` (see `jolt-core/src/zkvm/recursion/verifier.rs`):
  - `enable_gt_fused_end_to_end`
  - `enable_g1_scalar_mul_fused_end_to_end`
  - `enable_g1_fused_wiring_end_to_end`
  - `enable_g2_scalar_mul_fused_end_to_end`
  - `enable_g2_fused_wiring_end_to_end`

- **Stage 2 verifier toggles** (see `jolt-core/src/zkvm/recursion/verifier.rs`):
  - `JOLT_RECURSION_ENABLE_SHIFT_RHO` (default `true`)
  - `JOLT_RECURSION_ENABLE_SHIFT_G1_SCALAR_MUL` (default `true`)
  - `JOLT_RECURSION_ENABLE_SHIFT_G2_SCALAR_MUL` (default `true`)
  - `JOLT_RECURSION_ENABLE_PGX_REDUCTION` (default `true`)
  - `JOLT_RECURSION_ENABLE_WIRING`, `JOLT_RECURSION_ENABLE_WIRING_GT`, `..._G1`, `..._G2` (defaults `true`)
  - `JOLT_RECURSION_ENABLE_GT_FUSED_END_TO_END` (default `false`): enables fused GT (changes Stage-2 instance selection/challenge layout and Stage-3 packing layout).
  - `JOLT_RECURSION_ENABLE_G1_SCALAR_MUL_FUSED_END_TO_END` (default `false`): enables fused G1 scalar-mul (Stage 2 + Stage 3 packing).
  - `JOLT_RECURSION_ENABLE_G2_SCALAR_MUL_FUSED_END_TO_END` (default `false`): enables fused G2 scalar-mul (Stage 2 + Stage 3 packing).
  - `JOLT_RECURSION_ENABLE_G1_FUSED_WIRING_END_TO_END` (default `false`): enables fully fused G1 (scalar-mul + add + wiring; implies scalar-mul fusion).
  - `JOLT_RECURSION_ENABLE_G2_FUSED_WIRING_END_TO_END` (default `false`): enables fully fused G2 (scalar-mul + add + wiring; implies scalar-mul fusion).
  - `JOLT_RECURSION_STOP_AFTER_STAGE2` (tests only): early-exit hook to isolate Stage 2 failures.

- **Witness-generation toggles** (see `jolt-core/src/zkvm/recursion/witness_generation.rs`): enable/disable inclusion of specific constraint families
  (GT mul, G1/G2 scalar mul, G1/G2 add; pairing is feature-gated).

- **Cargo feature gates**:
  - `experimental-pairing-recursion`: enables experimental Multi-Miller-loop recursion gadgets.

---

## 9. Benchmarking & Cycle Measurement

This section describes how to measure cycle counts for the recursion verifier running inside the RISC-V guest.

### 9.1 Measuring Cycle Counts

To measure cycle counts without generating a full proof (which is slow), use the `trace` command with the `--committed` flag for committed program mode:

```bash
# Trace fibonacci recursion (committed program mode)
cargo run --release -p recursion -- trace --example fibonacci --embed --committed --cycle-tracking

# Trace muldiv recursion (committed program mode)
cargo run --release -p recursion -- trace --example muldiv --embed --committed --cycle-tracking

# With disk-based tracing (reduces memory usage for large traces)
cargo run --release -p recursion -- trace --example fibonacci --embed --committed --cycle-tracking --disk
```

### 9.2 Understanding the Output

The tracer outputs cycle counts for each instrumented section:

| Stage | Description |
|-------|-------------|
| `guest_verify_total` | Total guest verifier entrypoint (includes deserialization + verification) |
| `guest_one_proof_total` | Total work for one proof (includes deserialization + verification) |
| `guest_verify_one_proof_total` | Total verification for one proof (excluding deserialization) |
| `deserialize proof` | Deserializing the Jolt proof from bytes |
| `deserialize device` | Deserializing the I/O device |
| `deserialize preprocessing` | Deserializing verifier preprocessing |
| `deserialize recursion artifact` | Deserializing recursion artifact (if present) |
| `jolt_verify_stage1` - `stage7` | Jolt verifier stages (non-PCS) |
| `jolt_verify_stage8_dory_pcs` | Dory PCS verification (native) |
| `verify_recursion_total` | Recursion verification wrapper (replay base stages 1–7, derive recursion input, verify recursion SNARK, external pairing check) |
| `verify_recursion_base_stages_1_to_7_total` | Base verifier stages 1–7, replayed inside recursion verification |
| `verify_recursion_stage8_prep_total` | Stage 8 recursion prep (symbolic AST build + transcript replay + hint-based plan derivation) |
| `verify_recursion_snark_verify_total` | Recursion SNARK verification (stages 1–3 + PCS opening) |
| `jolt_external_pairing_check` | Final external pairing check (boundary value bound by wiring constraints) |
| `jolt_recursion_stage1` - `stage3` | Recursion SNARK stages (sumchecks + reductions) |
| `jolt_recursion_pcs_opening` | Recursion SNARK PCS opening verification |
| `jolt_hyrax_msm1`, `jolt_hyrax_msm2` | Hyrax multi-scalar multiplications |
| `jolt_hyrax_opening_verify` | Hyrax opening verification |

Each line shows both:
- **RV64IMAC cycles**: Raw RISC-V instruction cycles
- **Virtual cycles**: Cycles including memory operations and other costs

### 9.2.1 Flat totals vs nested drill-down (how to make spans “add up”)

Cycle markers are **nested** in many places (e.g. the Hyrax MSM spans are inside the Hyrax opening span, which is inside the recursion PCS-opening span, etc.). This is great for drill-down, but it means you **must not sum nested spans** (you will double count).

To make accounting easy, the verifier exposes a small set of **non-overlapping “flat totals”** with names ending in `*_total`. These are intended to be summed at a given level:

- **Guest-level total**: `guest_verify_total` should be very close to the tracer-reported `trace length`.
- **Per-proof**:
  - `guest_one_proof_total` = (deserialization for that proof) + `guest_verify_one_proof_total`
  - `guest_verify_one_proof_total` is the “verification-only” total for one proof (no deserialization).
- **Recursion wrapper** (when verifying with a recursion artifact):
  - `verify_recursion_total` \(\approx\)
    `verify_recursion_base_stages_1_to_7_total`
    + `verify_recursion_stage8_prep_total`
    + `verify_recursion_snark_verify_total`
    + `jolt_external_pairing_check`
    + (small overhead)
- **Recursion SNARK verifier**:
  - `verify_recursion_snark_verify_total` \(\approx\)
    `jolt_recursion_stage1` + `jolt_recursion_stage2` + `jolt_recursion_stage3` + `jolt_recursion_pcs_opening`
    + (small overhead: transcript/accumulator setup before Stage 1)

Practical rule of thumb:
- **For totals**: sum only the `*_total` spans (plus any other “flat” top-level spans you care about).
- **For hotspots**: use nested spans like `jolt_hyrax_msm1` / `jolt_hyrax_msm2` / `jolt_hyrax_opening_verify` to drill down inside `jolt_recursion_pcs_opening`, but do not sum them together with their parents.

### 9.3 Full Proof Generation & Verification

To generate proofs first, then verify them:

```bash
# Step 1: Generate proofs (committed program mode)
cargo run --release -p recursion -- generate --example fibonacci --committed

# Step 2: Verify proofs (with embedding for recursion)
cargo run --release -p recursion -- verify --example fibonacci --embed --committed
```

### 9.4 Comparing Program Modes

The `--committed` flag enables "committed program mode" where the verifier only receives commitments to the program (bytecode + program image) rather than the full program data. This is the mode used in production recursion.

```bash
# Without --committed: Full program mode (verifier gets full bytecode)
cargo run --release -p recursion -- trace --example fibonacci --embed

# With --committed: Committed program mode (verifier gets commitments only)
cargo run --release -p recursion -- trace --example fibonacci --embed --committed
```

---

## References

- [Jolt Paper](https://eprint.iacr.org/2023/1217)
- [Dory Paper](https://eprint.iacr.org/2020/1274)
- [Hyrax Paper](https://eprint.iacr.org/2017/1132)
- [Jagged Polynomial Commitments](https://eprint.iacr.org/2024/504) - Claim 3.2.1 for jagged transform
