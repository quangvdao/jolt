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
2. **Operation constraints (Stage 1)**: For every traced operation instance, we prove “this op is computed correctly in isolation”
   via a type-specific sumcheck (e.g., GT exp, GT mul, G1/G2 scalar mul, and (new) G1/G2 add).
3. **Wiring / copy constraints (Stage 1d)**: We prove that the output of each operation is exactly the input consumed by downstream
   operations, so the witness represents a single coherent computation DAG (not a bag of unrelated correct ops).
4. **Boundary outputs for the outside verifier**: The recursion SNARK exposes the **three (G1,G2) pairing input pairs** used by Dory’s
   final optimized check (a 3-way multi-pairing), and (optionally) the corresponding GT “rhs” value. The outside verifier then computes
   the multi-pairing and checks equality itself.

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
│  Stage 1: Constraint Sum-Checks                                     │
│  ─────────────────────────────                                      │
│  Prove validity of each op in isolation                              │
│  - GT Exp, GT Mul                                                    │
│  - G1/G2 ScalarMul                                                   │
│  - (NEW) G1/G2 Add                                                   │
│  Output: Virtual claims for inputs/outputs of each op               │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1b/c/d: Consistency Checks                                   │
│  ────────────────────────────────                                   │
│  1b: Shift Sumcheck (internal consistency of packed ops)            │
│  1c: Boundary Sumcheck (check initial/final states vs public input) │
│  1d: Wiring Sumcheck (AST-derived copy constraints)                 │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2: Virtualization Sum-Check                                  │
│  ─────────────────────────────────                                  │
│  Combine all claims into matrix M(s,x) → claim M(r_s, r_x)          │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3: Jagged Transform Sum-Check                                │
│  ────────────────────────────────────                               │
│  Sparse M → Dense q → claim q(r_dense)                              │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 4: Opening Proof (Hyrax over Grumpkin)                       │
│  ────────────────────────────────────────────                       │
│  Prove q(r_dense) = v_dense → final verification                    │
└─────────────────────────────────────────────────────────────────────┘
```

#### External step (outside this SNARK): final Dory multi-pairing check

After the recursion SNARK verifies, the outside verifier performs Dory’s final optimized check:
a **single 3-way multi-pairing** equality. The recursion SNARK provides the required three input pairs
\((p1\_g1,p1\_g2),(p2\_g1,p2\_g2),(p3\_g1,p3\_g2)\) as public outputs, and the outside verifier computes the
pairing product and compares it against the (public) GT rhs.

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

---

## 2. Stage 1: Constraint Sum-Checks

Stage 1 proves correctness of each expensive operation via constraint-specific sum-checks. Each produces virtual polynomial claims that feed into Stage 2.

### 2.1 Ring Switching & Quotient Technique

$\mathbb{G}_T$ elements are represented as $\mathbb{F}_{q^{12}} = \mathbb{F}_q[X]/(p(X))$ where $p(X)$ is an irreducible polynomial of degree 12.

**Key insight**: For $a, b, c \in \mathbb{G}_T$, the equation $a \cdot b = c$ holds iff there exists quotient $Q$ such that:
$$a(X) \cdot b(X) = c(X) + Q(X) \cdot p(X)$$

On the Boolean hypercube $\{0,1\}^4$ (viewing elements as 4-variate polynomials):
$$a(x) \cdot b(x) - c(x) - Q(x) \cdot g(x) = 0$$

where $g(x)$ is the MLE of $p$ on the hypercube.

This transforms high-degree $\mathbb{F}_{q^{12}}$ operations into low-degree constraints over $\mathbb{F}_q$ by introducing the quotient as auxiliary witness.

### 2.2 GT Exponentiation

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
- Total: 11 rounds, degree 4

#### Output Claims

After final challenges $(r_s^*, r_x^*)$:
- `PackedGtExpRho(i)`: $\rho(r_s^*, r_x^*)$
- `PackedGtExpRhoNext(i)`: $\rho_{\text{next}}(r_s^*, r_x^*)$
- `PackedGtExpQuotient(i)`: $Q(r_s^*, r_x^*)$

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
- Soundness error: Unchanged at $\text{deg}/|\mathbb{F}| = 4/p \approx 2^{-252}$
- The two-phase sumcheck maintains the same security as checking all packed step constraints (up to 128 base-4 steps)
- Batching with $\gamma$ preserves zero-knowledge properties

#### Performance Benefits

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Polynomials per GT exp | 1,024 | 3 | 341.3× |
| Virtual claims | 1,024 | 3 | 341.3× |
| Proof size contribution | ~32KB | ~96B | ~333× |

---

### 2.2.1 Stage 1b: Shift Sumcheck Optimization

The packed GT exponentiation can be further optimized by eliminating the `rho_next` polynomial commitment using a shift sumcheck protocol.

#### The Problem

Currently, we commit to three polynomials per packed GT exponentiation:
- `rho(s,x)` - intermediate values at each step
- `rho_next(s,x)` - shifted values where `rho_next(s,x) = rho(s+1,x)`
- `quotient(s,x)` - quotient polynomials

Since `rho_next` is completely determined by `rho` through the shift relationship, committing to it is redundant.

#### The Solution: Shift Sumcheck

Instead of committing to `rho_next`, we prove the shift relationship algebraically. After the constraint sumcheck completes at point `(r_s*, r_x*)`, we run an additional sumcheck to prove:

$$v = \sum_{s \in \{0,1\}^7} \sum_{x \in \{0,1\}^4} \text{EqPlusOne}(r_s^*, s) \cdot \text{eq}(r_x^*, x) \cdot \rho(s,x)$$

Where:
- `EqPlusOne(r_s*, s)` = 1 if `s = r_s* + 1`, 0 otherwise
- The sum evaluates to exactly `rho(r_s*+1, r_x*)` = `rho_next(r_s*, r_x*)`

#### Modified Protocol Flow

```
Stage 1a: Packed GT Constraint Sumcheck
  - Commits only to rho and quotient (NOT rho_next)
  - Outputs: virtual claims for rho, quotient
  - Outputs: unverified claim v = rho_next(r_s*, r_x*)
  ↓
Stage 1b: Shift Sumcheck (NEW)
  - Proves that v = rho(r_s*+1, r_x*)
  - 11 rounds, degree 2
  - Outputs: verified rho_next claim (used by packed GT exp correctness, boundary, and wiring)
  ↓
Stage 2: Direct Evaluation Protocol
  - Uses the matrix claims from Stage 1 (e.g., rho and quotient for packed GT exp). The verified rho_next claim may be consumed by consistency checks, but does not need to appear as a matrix row.
  - No changes needed
```

#### Accumulator Communication

The shift sumcheck integrates seamlessly with the existing virtual polynomial infrastructure:

```rust
// Stage 1a: After constraint sumcheck
accumulator.append_virtual(
    VirtualPolynomial::PackedGtExpRho(i),
    SumcheckId::PackedGtExp,
    (r_s_star, r_x_star),
    rho_claim,
);

// Stage 1b: After shift sumcheck verification
accumulator.append_virtual(
    VirtualPolynomial::PackedGtExpRhoNext(i),
    SumcheckId::ShiftSumcheck,  // Different sumcheck ID
    (r_s_star, r_x_star),
    verified_rho_next_claim,
);
```

Stage 2 treats both virtual claims uniformly, regardless of their verification method.

#### Benefits and Trade-offs

**Benefits**:
- Reduces committed polynomials from 5 to 4 per GT exp
- Saves ~160 bytes per GT exp in proof size
- Reduces memory requirements (no rho_next storage)

**Trade-offs**:
- Adds 11 sumcheck rounds
- Slightly more complex verifier (+24 field ops per round)
- Additional proof elements (+11 field elements)

**Security**: Maintains the standard sumcheck soundness bound \(O(\text{rounds}/|\mathbb{F}|)\) (here, rounds = 11).

---

### 2.3 GT Multiplication

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

- `RecursionMulLhs(i)`: $\tilde{a}_i(r_x')$
- `RecursionMulRhs(i)`: $\tilde{b}_i(r_x')$
- `RecursionMulResult(i)`: $\tilde{c}_i(r_x')$
- `RecursionMulQuotient(i)`: $\tilde{Q}_i(r_x')$

---

### 2.4 G1 Scalar Multiplication

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

The implementation uses 11 constraint variables (8 step variables, zero-padded to 11 for the
uniform Dory matrix layout). For each scalar-mul instance, constraints are batched with $\delta$,
and multiple instances are batched with $\gamma$:

$$
0 = \sum_{x \in \{0,1\}^{11}} \text{eq}(r_x, x)\cdot \sum_{i}\gamma^i\cdot\Big(\sum_{j}\delta^j\cdot C_{i,j}(x)\Big)
$$

- 11 rounds (one per constraint variable)
- Degree 6 overall (max constraint degree 5, times $\text{eq}(r_x,x)$ which is multilinear)

#### Output Claims

- `RecursionG1ScalarMulXA(i)`, `RecursionG1ScalarMulYA(i)` - accumulator coordinates
- `RecursionG1ScalarMulXT(i)`, `RecursionG1ScalarMulYT(i)` - doubled point coordinates
- `RecursionG1ScalarMulXANext(i)`, `RecursionG1ScalarMulYANext(i)` - next accumulator
- `RecursionG1ScalarMulTIndicator(i)` - 1 if T = O (infinity), 0 otherwise
- `RecursionG1ScalarMulAIndicator(i)` - 1 if A = O (infinity), 0 otherwise

---

### 2.5 G2 Scalar Multiplication

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

The sumcheck runs over 11 variables (8 step vars padded to 11 for the uniform Dory matrix layout).
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
0 = \sum_{x \in \{0,1\}^{11}} \text{eq}(r_x, x)\cdot \sum_{i}\gamma^i\cdot\Big(\sum_{j}\delta^j\cdot C_{i,j}(x)\Big)
$$

- 11 rounds
- Degree 6 overall (max term degree 5, times $\text{eq}(r_x,x)$)

#### Output Claims

All opened values are over $\mathbb{F}_q$ (components of $\mathbb{F}_{q^2}$):

- `RecursionG2ScalarMulXAC0(i)`, `RecursionG2ScalarMulXAC1(i)`
- `RecursionG2ScalarMulYAC0(i)`, `RecursionG2ScalarMulYAC1(i)`
- `RecursionG2ScalarMulXTC0(i)`, `RecursionG2ScalarMulXTC1(i)`
- `RecursionG2ScalarMulYTC0(i)`, `RecursionG2ScalarMulYTC1(i)`
- `RecursionG2ScalarMulXANextC0(i)`, `RecursionG2ScalarMulXANextC1(i)`
- `RecursionG2ScalarMulYANextC0(i)`, `RecursionG2ScalarMulYANextC1(i)`
- `RecursionG2ScalarMulTIndicator(i)`, `RecursionG2ScalarMulAIndicator(i)`

---

### 2.5.1 G1/G2 Addition (NEW)

Dory verification performs many explicit group additions in G1 and G2 (e.g., updates like
\(e1 \leftarrow e1 + \alpha \cdot X\), \(e2 \leftarrow e2 + \beta^{-1}\cdot Y\)).
In Dory’s AST these appear as `G1Add` / `G2Add` nodes, and they must be proven inside the recursion SNARK
because they affect the final pairing inputs.

We add two new Stage 1 sumchecks:

- **G1Add sumcheck**: proves that for each add instance, the output point equals \(P+Q\) in \(\mathbb{G}_1\), with correct
  handling of infinity, doubling, and inverse cases.
- **G2Add sumcheck**: same, but for \(\mathbb{G}_2\) over \(\mathbb{F}_{q^2}\), implemented by splitting \(\mathbb{F}_{q^2}\)
  coordinates into (c0,c1) components in \(\mathbb{F}_q\) and enforcing constraints component-wise.

The constraint system below is the **authoritative** description of what is implemented in:

- `jolt-core/src/zkvm/recursion/stage1/g1_add.rs`
- `jolt-core/src/zkvm/recursion/stage1/g2_add.rs`

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

All of the above are witness polynomials over the recursion Stage-1 domain \(\{0,1\}^{11}\)
(i.e., 11 variables / 2048 evaluations), matching the uniform Dory matrix layout.

**Why 11 variables (even if an op is “smaller”)?** Many underlying objects are naturally lower-variate:
- GT mul/field-element slices are naturally 4-var MLEs (size 16),
- scalar-mul traces are naturally 8-var MLEs (size 256),
- a single add instance’s ports are “morally” 0-var (just a single point).

In the recursion implementation, we embed all of them into a **common 11-var column domain** so that:
- Stage 1 can use a single shared opening point \(r_x^\*\) across all op types, and
- Stage 2/3 can treat every port polynomial uniformly as a row of the recursion matrix.

This embedding is done by **padding / gating** (typically zero-padding so the polynomial is supported only on a subcube),
so the additional “padding variables” do not increase the *logical* information content—only the *ambient* domain.
The jagged transform in Stage 3 later exploits this sparsity/jaggedness.

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
0 = \sum_{x\in\{0,1\}^{11}} \mathrm{eq}(r_x,x)\cdot\sum_{i}\gamma^i\cdot C_i(x).
\]

- Rounds: 11
- Degree bound: 6 (max constraint degree 5, times \(\mathrm{eq}(r_x,x)\) which is multilinear)

*Note:* “Rounds: 11” refers to this **ambient** 11-var embedding. A given witness polynomial may be zero-padded / independent
of some of those variables, but we still run the sumcheck over the full 11-var domain to match the matrix layout and shared
opening point.

#### Output claims (ports)

At the final Stage-1 evaluation point \(r_x^\*\), Stage 1 emits virtual claims for the input/output ports and auxiliaries:

- `RecursionG1AddXP(i)`, `RecursionG1AddYP(i)`, `RecursionG1AddPIndicator(i)`
- `RecursionG1AddXQ(i)`, `RecursionG1AddYQ(i)`, `RecursionG1AddQIndicator(i)`
- `RecursionG1AddXR(i)`, `RecursionG1AddYR(i)`, `RecursionG1AddRIndicator(i)`
- `RecursionG1AddLambda(i)`, `RecursionG1AddInvDeltaX(i)`
- `RecursionG1AddIsDouble(i)`, `RecursionG1AddIsInverse(i)`

These are the values Stage 1d wiring uses to connect add nodes to the rest of the AST DAG.

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

Concretely, the implementation in `stage1/g2_add.rs` enforces the following \(\mathbb{F}_q\) constraints
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

The sumcheck batching/round structure matches G1Add (11 rounds, degree bound 6), and Stage 1 emits the
corresponding virtual port claims at \(r_x^\*\) for Stage 1d wiring:

- `RecursionG2AddXPC0(i)`, `RecursionG2AddXPC1(i)`, `RecursionG2AddYPC0(i)`, `RecursionG2AddYPC1(i)`, `RecursionG2AddPIndicator(i)`
- `RecursionG2AddXQC0(i)`, `RecursionG2AddXQC1(i)`, `RecursionG2AddYQC0(i)`, `RecursionG2AddYQC1(i)`, `RecursionG2AddQIndicator(i)`
- `RecursionG2AddXRC0(i)`, `RecursionG2AddXRC1(i)`, `RecursionG2AddYRC0(i)`, `RecursionG2AddYRC1(i)`, `RecursionG2AddRIndicator(i)`
- `RecursionG2AddLambdaC0(i)`, `RecursionG2AddLambdaC1(i)`, `RecursionG2AddInvDeltaXC0(i)`, `RecursionG2AddInvDeltaXC1(i)`
- `RecursionG2AddIsDouble(i)`, `RecursionG2AddIsInverse(i)`

### 2.6 Stage 1c: Boundary Sumcheck

This stage enforces the initial and final states of the recursive operations.

#### Initial Conditions
- **GT Exp**: $\rho(0, x) = 1$ (Identity)
- **G1/G2 Scalar Mul**: $A(0) = \mathcal{O}$ (Infinity)

#### Final Conditions
- **GT Exp**: $\rho(N, x) = \text{result}(x)$
- **G1/G2 Scalar Mul**: $A(N) = \text{result}$

These are proven via a sumcheck over the step variable $s$, using `eq(0, s)` and `eq(N, s)` to isolate the boundary terms.

---

### 2.7 Stage 1d: Wiring (Copy) Constraints

This stage enforces that all Stage 1 operation instances form **one coherent computation DAG** (copy constraints),
rather than a multiset of unrelated valid operations.

#### Topology source: Dory’s `AstGraph`

Dory can record the full verification computation as an AST/DAG (`AstGraph`) where each node:
- produces a typed `ValueId` (G1/G2/GT),
- records an `AstOp` (e.g. `G1Add`, `G2ScalarMul`, `GTMul`, `GTExp`), and
- lists its input `ValueId`s (the data-flow edges).

Wiring constraints are derived **only** from this graph structure: for every node input `ValueId`, we add a copy constraint
from the producer’s output value to the consumer’s input port. This avoids trusting prover-supplied wiring metadata.

#### What is being wired: typed values (ports)

We wire **typed values** (G1/G2/GT elements) between operation instances:
- `GTExp` output → `GTMul` input
- `GTMul` output → downstream `GTMul` input
- `G1ScalarMul` output → `G1Add` input (this is common in Dory verifier updates like `e1 = e1 + alpha*X`)
- `G2ScalarMul` output → `G2Add` input
- `G1Add`/`G2Add` outputs → downstream inputs, etc.

This requires having sumchecks for the “primitive” ops that appear in Dory’s AST:
- existing: GT exp/mul, G1/G2 scalar mul
- **new**: G1 add, G2 add

#### Port extraction for packed traces: “gated slices”

Some operations are encoded as larger tables (e.g., packed GT exp, scalar-mul traces).
In those cases, a port (input/output value) is a **slice** of the table, extracted by multiplying by a selector polynomial:

- Example (conceptual): output of a 256-step scalar-mul trace is the accumulator at the last step, extracted via
  `Eq(step_last, step)` (“gated slice”).

This keeps witness size manageable and avoids creating a separate polynomial per step.

#### Wiring sumcheck (Stage 1d)

For each type (G1/G2/GT), we run a dedicated wiring sumcheck that:
1. samples a fresh random evaluation point \(r\) (Fiat–Shamir),
2. evaluates the relevant port expressions at \(r\), and
3. checks a single random linear combination of all copy constraints equals 0.

At a high level, for edges \(e\) we check:
\[
0 \stackrel{!}{=} \sum_{e} \lambda_e \cdot (\text{PortOut}_{\text{src}(e)}(r) - \text{PortIn}_{\text{dst}(e)}(r)),
\]
where \(\lambda_e\) are transcript challenges to prevent cancellation.

This design **does not rely** on Stage 1 instances sharing a common opening point; Stage 1d introduces its own shared point(s)
for wiring.

#### Connection to the external pairing boundary

The Dory verifier’s final optimized check is a **3-way multi-pairing** (see `third-party/dory/src/reduce_and_fold.rs::DoryVerifierState::verify_final`).
We do not prove pairings inside this SNARK. Instead:
- the recursion witness includes all non-pairing operations that produce the three pairing input pairs, and
- the recursion SNARK exposes those three (G1,G2) pairs as public outputs,
so the outside verifier can perform the final pairing check.

---

## 3. Stage 2: Direct Evaluation Protocol

After Stage 1, we have many virtual polynomial claims $(v_0, v_1, \ldots, v_{n-1})$ at point $r_x$. Stage 2 verifies these claims directly without sumcheck.

### 3.1 Why Direct Evaluation

The constraint matrix $M$ has a special structure:
$$M(i, r_x) = v_i \text{ for all } i$$

Therefore:
$$M(r_s, r_x) = \sum_{i \in \{0,1\}^{\log n}} \text{eq}(r_s, i) \cdot M(i, r_x) = \sum_{i} \text{eq}(r_s, i) \cdot v_i$$

This allows direct computation without sumcheck rounds.

### 3.2 Matrix Organization

Define matrix $M : \{0,1\}^s \times \{0,1\}^x \to \mathbb{F}_q$ where:
- Row index $s$ selects polynomial type and constraint index
- Column index $x$ selects evaluation point (4 or 8 bits)

Row indexing:
```
row = poly_type × num_constraints_padded + constraint_idx
```

Polynomial types:

The sparse matrix rows correspond to the set of “virtual polynomials” emitted by Stage 1 for all supported op types.
This includes:

- GT ops: GT mul ports and (packed) GT exp committed polynomials (e.g., \(\rho\) and \(Q\))
- G1 scalar mul ports (x/y for \(A,T,A'\) plus infinity indicators)
- G2 scalar mul ports (same, with Fq2 split into c0/c1 components)
- (NEW) G1 add / G2 add ports (for explicit add nodes in Dory’s AST)

**Implementation note**: The exact enumeration and ordering is an implementation detail of the recursion constraint system
(see `jolt-core/src/zkvm/recursion/constraints_sys.rs`), and should be treated as authoritative over any hard-coded table here.

### 3.3 Direct Evaluation Protocol

**Input**: Virtual claims $\{v_i\}$ from Stage 1 at point $r_x$

**Protocol**:
1. **Sample**: $r_s \leftarrow \mathbb{F}^{\log n}$ from transcript
2. **Prover**:
   - Evaluate $M(r_s, r_x)$ by binding matrix to challenges
   - Send evaluation to verifier
3. **Verifier**:
   - Compute $\text{eq}_{\text{evals}} = \text{EqPolynomial::evals}(r_s)$
   - Compute $v = \sum_i \text{eq}_{\text{evals}}[s_i] \cdot v_i$ where $s_i = \text{matrix\_s\_index}(i)$
   - Verify prover's evaluation matches $v$

**Output**: Opening claim $M(r_s, r_x) = v_{\text{sparse}}$

#### Mathematical Correctness

**Theorem**: Direct evaluation is sound with the same security as sumcheck.

**Proof**: The multilinear extension $\tilde{M}$ is unique. For the boolean hypercube:
$$\tilde{M}(s, x) = M(s, x) \text{ for all } s \in \{0,1\}^{\log n}, x$$

By linearity of multilinear extensions:
$$\tilde{M}(r_s, r_x) = \sum_{i \in \{0,1\}^{\log n}} M(i, r_x) \cdot \text{eq}(r_s, i)$$

Since $M(i, r_x) = v_i$ by construction:
$$\tilde{M}(r_s, r_x) = \sum_i v_i \cdot \text{eq}(r_s, i)$$

The verifier computes exactly this sum, so the protocol is perfectly sound. □

**Security Guarantees**:
- No additional soundness error (deterministic protocol)
- Fiat-Shamir security maintained through transcript inclusion
- Binding: Prover committed to $v_i$ values in Stage 1

**Benefits**:
- Eliminates $\log n$ sumcheck rounds
- Reduces proof size by $3\log n$ field elements
- Verifier work remains $O(n)$

---

## 4. Stage 3: Jagged Transform Sum-Check

The matrix $M$ is sparse: 4-variable polynomials (GT) are zero-padded to 8 variables. Stage 3 compresses to a dense representation.

### 4.1 Why Jaggedness

| Operation | Native Variables | Native Size | Padded Size |
|-----------|-----------------|-------------|-------------|
| GT Exp/Mul | 4 | 16 | 256 (zero-padded) |
| G1 Scalar Mul | 8 | 256 | 256 (native) |

The jagged transform eliminates redundant zeros, reducing commitment size.

### 4.2 Sparse-to-Dense Bijection

**Sparse**: $M(s, x)$ with row $s$ and column $x \in \{0,1\}^8$

**Dense**: $q(i)$ containing only non-redundant entries

Bijection:
- `row(i)` = polynomial index for dense index $i$
- `col(i)` = evaluation index within that polynomial
- Cumulative sizes track where each polynomial's entries begin

### 4.3 Differences from Jagged PCS Paper

| Paper | Our Implementation |
|-------|-------------------|
| Column-wise jaggedness | Row-wise jaggedness |
| Direct bijection to matrix | Three-level mapping |
| Simple row/col | poly_idx → (constraint_idx, poly_type) → matrix_row |

The three-level mapping handles multiple polynomial types per constraint.

### 4.4 The Jagged Indicator Function

$$\hat{f}_{\text{jagged}}(r_s, r_x, r_{\text{dense}}) = \sum_{y \in \text{rows}} \text{eq}(r_s, y) \cdot \hat{g}(r_x, r_{\text{dense}}, t_{y-1}, t_y)$$

where $g(a, b, c, d) = 1$ iff $b < d$ AND $b = a + c$.

### 4.5 Branching Program Optimization

The function $\hat{g}$ can be computed in $O(n)$ time via a width-4 read-once branching program:
- Tracks carry bit (addition check: $b = a + c$)
- Tracks comparison bit ($b < d$)
- Processes bits LSB to MSB

This avoids naive $O(2^{4n})$ computation.

### 4.6 Sum-Check Protocol

**Input**: $M(r_{s,\text{final}}, r_x) = v_{\text{sparse}}$ from Stage 2

**Relation**:
$$v_{\text{sparse}} = \sum_{i \in \{0,1\}^{\ell_{\text{dense}}}} q(i) \cdot \hat{f}_{\text{jagged}}(r_s, r_x, i)$$

**Protocol**:
1. Prover materializes dense $q(i)$ and indicator $\hat{f}_{\text{jagged}}$
2. Sum-check over $\ell_{\text{dense}}$ rounds (degree 2)
3. At challenge $r_{\text{dense}}$: claim $q(r_{\text{dense}}) = v_{\text{dense}}$
4. Verifier computes $\hat{f}_{\text{jagged}}(r_s, r_x, r_{\text{dense}})$ via branching program
5. Verify: output = $v_{\text{dense}} \cdot \hat{f}_{\text{jagged}}$

**Output**: $(q, r_{\text{dense}}, v_{\text{dense}})$

### 4.7 Stage 3b: Jagged Assist

After Stage 3's sumcheck, the verifier must compute:
$$\hat{f}_{\text{jagged}}(r_s, r_x, r_{\text{dense}}) = \sum_{y \in [K]} \text{eq}(r_s, y) \cdot \hat{g}(r_x, r_{\text{dense}}, t_{y-1}, t_y)$$

where $K$ is the number of polynomials. For large systems, this requires $K \times O(\text{bits})$ field operations.

#### The Jagged Assist Protocol

Instead of computing all $K$ evaluations, we use batch verification:

1. **Prover sends**: $v_y = \hat{g}(r_x, r_{\text{dense}}, t_{y-1}, t_y)$ for all $y \in [K]$

2. **Verifier samples**: Batching coefficient $r \leftarrow \mathbb{F}$

3. **Batch claim**:
   $$\sum_{y=0}^{K-1} r^y \cdot \hat{g}(x_y) = \sum_{y=0}^{K-1} r^y \cdot v_y$$

4. **Sumcheck**: Over $P(b) = g(b) \cdot \sum_y r^y \cdot \text{eq}(b, x_y)$

5. **Final verification**: At random point $\rho$:
   - Compute $\hat{g}(\rho)$ using branching program
   - Compute $\sum_y r^y \cdot \text{eq}(\rho, x_y)$
   - Verify $P(\rho) = \hat{g}(\rho) \cdot \text{eq\_sum}$

#### Mathematical Correctness

**Theorem (Jagged Assist Soundness)**: If $\exists y^* : v_{y^*} \neq \hat{g}(x_{y^*})$, then the verifier accepts with probability at most $\frac{2m}{|\mathbb{F}|}$.

**Proof Sketch**:
1. Define polynomial $R(Z) = \sum_{y=0}^{K-1} Z^y \cdot (\hat{g}(x_y) - v_y)$
2. If any $v_{y^*}$ is incorrect, then $R(Z) \not\equiv 0$ and $\deg(R) \leq K-1$
3. By Schwartz-Zippel, $\Pr[R(r) = 0] \leq \frac{K-1}{|\mathbb{F}|}$
4. If $R(r) \neq 0$, the sumcheck polynomial $P$ differs from the honest polynomial
5. Sumcheck soundness: $\Pr[\text{accept} | R(r) \neq 0] \leq \frac{2m}{|\mathbb{F}|}$
6. Union bound: $\Pr[\text{accept}] \leq \frac{K-1 + 2m}{|\mathbb{F}|} \approx \frac{2m}{|\mathbb{F}|}$

**Key Insight**: The protocol converts $K$ individual checks into one batched sumcheck with negligible soundness loss.

#### Benefits

| Metric | Without Assist | With Assist | Improvement |
|--------|----------------|-------------|-------------|
| Verifier ops | $K \times 1,300$ | $330K$ | ~64× |
| Extra rounds | 0 | $2m$ | Acceptable |
| Soundness error | 0 | $\frac{2m}{|\mathbb{F}|}$ | Negligible |

The protocol leverages Lemma 4.6 (forward-backward decomposition) for efficient prover computation.

---

## 5. Stage 4: Opening Proof (Hyrax over Grumpkin)

Stage 4 proves the dense polynomial claim using Hyrax.

### 5.1 Why Hyrax

- **No pairing required**: Works over any curve with efficient MSM
- **Grumpkin-native**: Scalar field = $\mathbb{F}_q$, matching our constraint field
- **Square matrix efficiency**: Optimal for virtualized polynomials

### 5.2 Commitment Structure

For polynomial $q$ with $2^n$ evaluations:
1. Reshape as $2^{n_r} \times 2^{n_c}$ matrix
2. Commit to each row via Pedersen over Grumpkin
3. Commitment: $C = \{C_0, \ldots, C_{2^{n_r}-1}\}$

### 5.3 Opening Protocol

**Input**: Commitment $C$, point $r_{\text{dense}}$, claimed value $v_{\text{dense}}$

**Protocol**:
1. Decompose point into row/column components
2. Sum-check proves tensor product structure
3. Column opening proves consistency
4. Verifier checks against commitment

### 5.4 Final Verification

The verifier accepts iff all checks pass:
- Stage 1-3 sum-check verifications
- Hyrax opening verification

---

## 6. Parameters & Cost Analysis

This section provides analytical formulas for proof sizes, constraint counts, and computational costs.

### 6.1 Sumcheck Degrees and Rounds

| Stage | Protocol | Degree | Rounds | Elements/Round |
|-------|----------|--------|--------|----------------|
| 1 | GT Exponentiation (unpacked) | 4 | 4 | 5 |
| 1 | GT Exponentiation (packed) | 4 | 11 | 5 |
| 1 | GT Multiplication | 3 | 4 | 4 |
| 1 | G1 Scalar Multiplication | 6 | $\ell$ | 7 |
| 2 | Direct Evaluation | - | 0 | 1 |
| 3 | Jagged Transform | 2 | $d$ | 3 |
| 3b | Jagged Assist | 2 | $2m$ | 3 |

Where:
- $\ell = \lceil \log_2 n \rceil$ for $n$-bit scalar
- $d = \lceil \log_2(\text{dense\_size}) \rceil$
- $m = $ number of jagged indicator evaluations (typically $\log K$ where $K$ is polynomial count)

### 6.2 Constraint Counts

**Per-operation constraints** (high level):

- **GT Exp (unpacked, \(t\)-bit)**: \(t\) constraints, 4 polynomial types per bit (base, rho-prev, rho-curr, quotient).
- **GT Exp (packed, base-4)**: 1 packed constraint covering up to 128 base-4 steps, with 3 packed witness tables
  \((\rho,\rho_{\text{next}},Q)\) (plus additional public-input-derived tables if materialized).
- **GT Mul**: 1 constraint, 4 polynomial types (lhs, rhs, result, quotient).
- **G1 Scalar Mul (256-bit)**: 1 constraint over an 11-var domain (8 step bits padded to 11), with accumulator/double/next coordinates plus indicators.
- **G2 Scalar Mul (256-bit)**: same shape as G1 but with Fq2 split into (c0,c1) components and additional constraints.
- **G1Add / G2Add (new)**: 1 constraint per add node (with the exact polynomial interface defined by the add sumcheck implementation).

**Implementation note**: The exact set of “poly types” that become matrix rows is defined by the recursion constraint system
(`jolt-core/src/zkvm/recursion/constraints_sys.rs`) and is the authoritative source of truth.

### 6.3 Matrix Dimensions

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

### 6.4 Dense Size Computation

The jagged transform compresses the sparse matrix by removing zero-padding:

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

$\Rightarrow d = \lceil \log_2 2112 \rceil = 12$ rounds for Stage 3.

### 6.5 Proof Size Formulas

**Stage 1** (sumcheck messages):
$$|P_1| = \sum_{\text{type } t} (\text{degree}_t + 1) \times \text{rounds}_t$$

| Type | Degree | Rounds | Elements |
|------|--------|--------|----------|
| GT Exp (unpacked) | 4 | 4 | 20 |
| GT Exp (packed) | 4 | 12 | 60 |
| GT Mul | 3 | 4 | 16 |
| G1 Scalar Mul | 6 | $\ell$ | $7\ell$ |

**Stage 2** (direct evaluation):
$$|P_2| = 1 \text{ element}$$

**Stage 3** (jagged transform):
$$|P_3| = 3d \text{ elements}$$

**Stage 3b** (jagged assist):
$$|P_{3b}| = K + 3(2m) \text{ elements}$$
where $K$ is the number of $\hat{g}$ evaluations sent

**Stage 4** (Hyrax opening):
$$|P_4| = O(\sqrt{\text{dense\_size}}) \text{ group elements}$$

**Total proof size** (field elements, excluding PCS):
$$|P| = |P_1| + |P_2| + |P_3| + |P_{3b}| + \text{virtual claims}$$

### 6.6 Concrete Example: Single 256-bit GT Exponentiation (Unpacked vs Packed)

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

### 6.7 Prover Complexity

| Stage | Operation | Complexity |
|-------|-----------|------------|
| Stage 1 | Sumcheck per round | $O(2^{\text{vars}} \cdot \text{degree})$ |
| Stage 2 | Virtualization | $O(2^s)$ |
| Stage 3 | Jagged transform | $O(\text{dense\_size})$ |
| Stage 4 | Hyrax commitment | $O(\text{dense\_size})$ MSMs |

**Dominant cost**: Stage 1 sumcheck computation scales with constraint count.

**Parallelization**: Stage 1's three constraint types run in parallel.

### 6.8 Verifier Complexity

| Stage | Operation | Complexity |
|-------|-----------|------------|
| Stage 1 | Sumcheck verification | $O(\text{rounds} \cdot \text{degree})$ |
| Stage 2 | Claim aggregation | $O(c)$ |
| Stage 3 | Branching program | $O(\text{num\_polys} \cdot \text{bits})$ |
| Stage 4 | Hyrax verification | $O(\sqrt{\text{dense\_size}})$ |

**Key efficiency**: Stage 3 verifier uses $O(n)$ branching program instead of naive $O(2^{4n})$ MLE evaluation.

### 6.9 Comparison: Sparse vs Dense

Without jagged transform (sparse):
- Matrix size: $15 \cdot c_{\text{pad}} \times 256$
- For 256 GT exp constraints: $3,840 \times 256 = 983,040$ entries

With jagged transform (dense):
- Dense size: $1,024 \times 16 = 16,384$ entries
- **Compression ratio**: $60\times$

This compression directly reduces:
- PCS commitment size
- Stage 3 prover work
- Hyrax opening proof size

### 6.10 Scaling Summary

| Metric | Formula | 256-bit GT Exp |
|--------|---------|----------------|
| Constraints | $t$ (bit-length) | 256 |
| Stage 1 rounds | 4 (fixed for GT) | 4 |
| Stage 2 rounds | $\lceil \log_2(15c) \rceil$ | 12 |
| Stage 3 rounds | $\lceil \log_2(4tc \cdot 16) \rceil$ | 14 |
| Proof elements | $O(c + \log c)$ | ~1,100 |
| Prover time | $O(c \cdot 2^4)$ | ~4,000 ops |
| Verifier time | $O(c + \text{polys} \cdot \text{bits})$ | ~15,000 ops |

The protocol achieves **logarithmic scaling** in proof size relative to constraint count, with linear prover work and near-linear verifier work.

### 6.11 Unified Polynomial Cost Analysis

The packed GT exponentiation optimization dramatically reduces system costs:

**Memory Requirements**:

| Approach | Polynomials | Memory (256-bit exp) | Formula |
|----------|-------------|---------------------|---------|
| Unpacked | $4t$ | ~32 MB | $4t \times 2^4 \times 32$ bytes |
| Packed | 3 | ~0.2 MB | $3 \times 2^{11} \times 32$ bytes |

**Stage 2 Virtual Claims**:

| Approach | Virtual Claims | Verifier Work |
|----------|---------------|---------------|
| Unpacked | 1,024 | $O(1024)$ field ops |
| Packed | 3 | $O(3)$ field ops |

**Impact on Later Stages**:
- Stage 3 processes fewer polynomials (3 vs 1,024)
- Stage 3b benefits from reduced $K$ in batch verification
- Hyrax commitment is more efficient with fewer polynomials

**Trade-offs**:
- Packed approach uses 11-round sumcheck vs 4-round
- Slightly larger Stage 1 proof (60 vs 20 elements)
- Prover computes over larger domain ($2^{11}$ vs $2^4$)
- Net benefit: ~9.5× smaller total proof, ~200× fewer virtual claims

---

## 7. Implementation

This section describes the code architecture and data flow for the recursion implementation.

### 7.1 Module Structure

```
jolt-core/src/zkvm/recursion/
├── mod.rs                    # Re-exports and module docs
├── witness.rs                # Witness types (GTExpWitness, GTMulWitness, G1ScalarMulWitness)
├── constraints_sys.rs        # DoryMatrixBuilder, DoryMultilinearMatrix, ConstraintSystem
├── bijection.rs              # VarCountJaggedBijection, JaggedTransform trait
├── recursion_prover.rs       # RecursionProver orchestrating all stages
├── recursion_verifier.rs     # RecursionVerifier
├── stage1/
│   ├── packed_gt_exp.rs         # Packed GT exponentiation sumcheck (base-4 digits)
│   ├── shift_rho.rs             # Shift consistency for packed GT exp (Stage 1b)
│   ├── gt_mul.rs                # GT multiplication sumcheck
│   ├── g1_scalar_mul.rs         # G1 scalar mul sumcheck
│   ├── g2_scalar_mul.rs         # G2 scalar mul sumcheck
│   ├── g1_add.rs                # (NEW) G1 addition sumcheck
│   ├── g2_add.rs                # (NEW) G2 addition sumcheck
│   ├── boundary_constraints.rs  # Initial/final state checks (Stage 1c)
│   └── wiring.rs                # AST-derived wiring/copy constraints (Stage 1d)
├── stage2/
│   └── virtualization.rs        # Direct evaluation protocol
└── stage3/
    ├── jagged.rs                # Jagged transform sumcheck
    └── branching_program.rs     # O(n) MLE evaluation for g function
```

### 7.2 The Offloading Pattern

The revised design treats the **Dory proof** as the ground truth and does **not** rely on a “HintMap” that supplies intermediate
group elements to the verifier.

The offloading boundary is:

- **Inside the recursion SNARK**: all non-pairing group/GT computation implied by Dory verification (scalar mul, add, GT exp/mul, wiring).
- **Outside the recursion SNARK**: transcript hashing / Fiat–Shamir challenge derivation, scalar-field arithmetic (inverses/products), and the
  **final 3-way multi-pairing check**.

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

### 7.3 Witness Types

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
struct PackedGtExpWitness {
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

### 7.4 Constraint System Construction

The recursion constraint system is constructed from:

- the extracted `WitnessCollection` (concrete witness traces), and
- the extracted `AstGraph` (the authoritative operation DAG / topology).

Conceptually, we traverse the AST nodes and:
- allocate a constraint instance for each operation node we prove inside the SNARK (GT exp/mul, G1/G2 scalar mul, G1/G2 add),
- attach the corresponding witness polynomials from `WitnessCollection` (for traced ops), and
- attach any needed public inputs (setup/proof elements and transcript-derived scalars).

```rust
let mut builder = DoryMatrixBuilder::new();

// Pseudocode: iterate AST nodes in topological order
for node in ast.nodes {
    match node.op {
        AstOp::GTExp { .. } => builder.add_packed_gt_exp_witness(...),
        AstOp::GTMul { .. } => builder.add_gt_mul_witness(...),
        AstOp::G1ScalarMul { .. } => builder.add_g1_scalar_mul_witness(...),
        AstOp::G2ScalarMul { .. } => builder.add_g2_scalar_mul_witness(...),
        AstOp::G1Add { .. } => builder.add_g1_add_witness(...),
        AstOp::G2Add { .. } => builder.add_g2_add_witness(...),
        _ => { /* inputs, negations, pairings handled elsewhere */ }
    }
}

let constraint_system: ConstraintSystem = builder.build();
```

The resulting `DoryMultilinearMatrix` has shape:

```
M : {0,1}^num_s_vars × {0,1}^num_x_vars → Fq

where:
  num_s_vars = ⌈log₂(15 × num_constraints_padded)⌉
  num_x_vars = 8  (max of 4-var GT and 8-var G1 polynomials)
```

### 7.5 Prover Flow

```rust
impl RecursionProver {
    pub fn prove(
        constraint_system: &ConstraintSystem,
        witnesses: &WitnessCollection,
        transcript: &mut Transcript,
    ) -> RecursionProof {
        // Stage 1: Run sumchecks in parallel
        let stage1_provers = vec![
            GtExpProver::new(...),
            GtMulProver::new(...),
            G1ScalarMulProver::new(...),
            G2ScalarMulProver::new(...),
            G1AddProver::new(...),
            G2AddProver::new(...),
        ];
        let (stage1_proof, stage1_claims) = BatchedSumcheck::prove(
            stage1_provers,
            &mut accumulator,
            transcript,
        );

        // Stage 2: Virtualization
        let stage2_prover = RecursionVirtualizationProver::new(
            constraint_system,
            &stage1_claims,
            transcript,
        );
        let (stage2_proof, stage2_claim) = stage2_prover.prove(...);

        // Stage 3: Jagged transform
        let stage3_prover = JaggedSumcheckProver::new(
            &bijection,
            &stage2_claim,
            transcript,
        );
        let (stage3_proof, dense_claim) = stage3_prover.prove(...);

        // PCS opening
        let opening_proof = accumulator.prove_openings(pcs, transcript);

        RecursionProof { stage1_proof, stage2_proof, stage3_proof, opening_proof }
    }
}
```

### 7.6 Verifier Flow

```rust
impl RecursionVerifier {
    pub fn verify(
        proof: &RecursionProof,
        verifier_input: &RecursionVerifierInput,
        transcript: &mut Transcript,
    ) -> Result<()> {
        // Stage 1: Verify batched sumcheck
        let stage1_verifiers = vec![
            GtExpVerifier::new(...),
            GtMulVerifier::new(...),
            G1ScalarMulVerifier::new(...),
            G2ScalarMulVerifier::new(...),
            G1AddVerifier::new(...),
            G2AddVerifier::new(...),
        ];
        let stage1_claims = BatchedSumcheck::verify(
            stage1_verifiers,
            &proof.stage1_proof,
            &mut accumulator,
            transcript,
        );

        // Stage 2: Verify virtualization
        let stage2_verifier = RecursionVirtualizationVerifier::new(...);
        let stage2_claim = stage2_verifier.verify(&proof.stage2_proof, ...)?;

        // Stage 3: Verify jagged transform
        // Verifier computes f_jagged using branching program (O(n) time)
        let stage3_verifier = JaggedSumcheckVerifier::new(
            &verifier_input.bijection,
            &verifier_input.matrix_rows,  // Precomputed
        );
        let dense_claim = stage3_verifier.verify(&proof.stage3_proof, ...)?;

        // PCS verification
        accumulator.verify_openings(pcs, &proof.opening_proof, transcript)?;

        Ok(())
    }
}
```

### 7.7 Opening Accumulator

The `OpeningAccumulator` tracks virtual polynomial claims across stages:

```rust
// Stage 1 appends virtual claims
accumulator.append_virtual(
    VirtualPolynomial::Recursion(RecursionPoly::GtExp { term: GtExpTerm::Rho, instance: constraint_idx }),
    SumcheckId::GtExp,
    opening_point,
    claimed_value,
);

// Stage 2 reads Stage 1 claims
let (point, value) = accumulator.get_virtual_polynomial_opening(
    VirtualPolynomial::Recursion(RecursionPoly::GtExp { term: GtExpTerm::Rho, instance: i }),
    SumcheckId::GtExp,
);

// Stage 3 outputs committed polynomial claim
accumulator.append_committed(
    CommittedPolynomial::DenseMatrix,
    opening_point,
    claimed_value,
);
```

### 7.8 Jagged Bijection

The `VarCountJaggedBijection` maps between sparse and dense indices:

```rust
impl JaggedTransform for VarCountJaggedBijection {
    /// Given dense index i, return which polynomial it belongs to
    fn row(&self, dense_idx: usize) -> usize {
        // Binary search in cumulative_sizes
        self.cumulative_sizes.partition_point(|&s| s <= dense_idx)
    }

    /// Given dense index i, return offset within that polynomial
    fn col(&self, dense_idx: usize) -> usize {
        let poly_idx = self.row(dense_idx);
        dense_idx - self.cumulative_sizes[poly_idx]
    }

    /// Map (poly_idx, eval_idx) to dense index, if valid
    fn sparse_to_dense(&self, poly_idx: usize, eval_idx: usize) -> Option<usize> {
        if eval_idx < self.poly_sizes[poly_idx] {
            Some(self.cumulative_sizes[poly_idx] + eval_idx)
        } else {
            None  // Padded region
        }
    }
}
```

### 7.9 Matrix Row Mapping

The three-level mapping converts polynomial indices to matrix rows:

```rust
// Level 1: Dense index → Polynomial index (via bijection)
let poly_idx = bijection.row(dense_idx);

// Level 2: Polynomial index → (constraint_idx, poly_type)
let (constraint_idx, poly_type) = mapping.decode(poly_idx);

// Level 3: (constraint_idx, poly_type) → Matrix row
let matrix_row = poly_type as usize * num_constraints_padded + constraint_idx;
```

This mapping is precomputed for the verifier:

```rust
let matrix_rows: Vec<usize> = (0..num_polynomials)
    .map(|poly_idx| {
        let (constraint_idx, poly_type) = mapping.decode(poly_idx);
        poly_type as usize * num_constraints_padded + constraint_idx
    })
    .collect();
```

### 7.10 Polynomial Type Enumeration

The recursion matrix row layout is an encoding detail of the constraint system and evolves as we add new op types (e.g., G1/G2 add)
and refine packed protocols (e.g., packed GT exp). The authoritative definition is the `PolyType` enum in:

- `jolt-core/src/zkvm/recursion/constraints_sys.rs`

### 7.11 Dory Integration: Witness + AST Extraction (No HintMap)

We run Dory verification once (prover-side) with a trace context that records:

- a **WitnessCollection**: per-op witness traces for Stage 1 sumchecks, and
- an **AstGraph**: the full computation DAG used to derive wiring/copy constraints deterministically.

Dory supports enabling AST tracing via `TraceContext::for_witness_gen_with_ast()` and extracting both artifacts via `finalize_with_ast()`.

```rust
let ctx = Rc::new(TraceContext::<W, E, Gen>::for_witness_gen_with_ast());
verify_recursive(commitment, evaluation, point, proof, setup, transcript, ctx.clone())?;
let (witnesses_opt, ast_opt) = Rc::try_unwrap(ctx).ok().unwrap().finalize_with_ast();
```

**Verifier-side topology**: the recursion verifier must not trust a prover-supplied AST. The intended design is that the recursion verifier
reconstructs the same AST skeleton deterministically from public inputs (Dory proof, setup, and transcript-derived scalars), without performing
any expensive group operations.

### 7.12 GPU Considerations

The recursion prover has a clear separation between orchestration logic (Rust) and compute-intensive kernels (GPU).

#### GPU-Accelerated Components

| Component | Operation | Why GPU |
|-----------|-----------|---------|
| Stage 1 Sumcheck | Polynomial evaluations over hypercube | Parallel evaluation of $2^n$ points |
| Stage 2 Direct Eval | Direct polynomial evaluation | Efficient dot product computation |
| Stage 3 Sumcheck | Jagged transform sumcheck | Dense polynomial operations |
| Hyrax Commit | Multi-scalar multiplication (MSM) | $O(\sqrt{N})$ group operations |
| Hyrax VMP | Vector-matrix product | Parallelizable linear algebra |

#### Rust-Side Orchestration

The following remain on the CPU/Rust side:

- **Witness generation**: `TraceContext` collection during Dory verify
- **Transcript management**: Fiat-Shamir challenge generation
- **Bijection logic**: `VarCountJaggedBijection` index computations
- **Matrix row mapping**: Three-level decode (poly_idx → constraint_idx → matrix_row)
- **AST extraction**: `AstGraph` extraction and (verifier-side) deterministic reconstruction for wiring
- **Proof assembly**: Collecting stage outputs into `RecursionProof` (including the external pairing boundary outputs)

#### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RUST (CPU)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  dory_verify_trace() ──→ (WitnessCollection, AstGraph) ──→ ConstraintSystem │
│                                              │                          │
│  Transcript ←─────────────────────────────── │ ←── challenges           │
│       │                                      │                          │
│       ▼                                      ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         GPU KERNELS                              │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  Stage 1: sumcheck_prove(polys, eq_evals) → round_polys         │   │
│  │  Stage 2: direct_eval(virtual_claims, eq_evals) → evaluation    │   │
│  │  Stage 3: sumcheck_prove(dense, f_jagged) → round_polys         │   │
│  │  Hyrax:   msm(scalars, bases) → commitments                     │   │
│  │           vmp(matrix, vector) → result                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  RecursionProof { stage1, stage2, stage3, opening }                    │
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

The jagged transform (Stage 3) reduces GPU memory pressure by ~60× compared to operating on the sparse matrix directly.

---

## References

- [Jolt Paper](https://eprint.iacr.org/2023/1217)
- [Dory Paper](https://eprint.iacr.org/2020/1274)
- [Hyrax Paper](https://eprint.iacr.org/2017/1132)
- [Jagged Polynomial Commitments](https://eprint.iacr.org/2024/504) - Claim 3.2.1 for jagged transform
