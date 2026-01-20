# G1 Scalar Multiplication Sumcheck Specification (BN254)

This document specifies the **implemented** Stage-1 scalar-multiplication constraints in
`jolt-core/src/zkvm/recursion/stage1/g1_scalar_mul.rs`.

Important: the previous version of this file claimed “complete soundness” for a constraint set
that **did not match the implementation** (and also contained incorrect constraints). This version
is aligned to the code and is careful about what is (and is not) enforced.

## Overview

We prove correctness of a double-and-add *transition relation* for 256 steps of a scalar
multiplication in BN254 G1.

- The scalar bits are treated as **public inputs** (derived from the scalar `Fr`), so we do **not**
  commit to a bit polynomial and we do **not** include a bit-booleanity constraint.
- Infinity is encoded as affine coordinates `(0,0)` plus an indicator bit.

## Field / Curve

- Base field: `Fq = ark_bn254::Fq` (prime field)
- Scalar field: `Fr = ark_bn254::Fr`
- Curve equation (BN254 G1): \(y^2 = x^3 + 3\) over `Fq`.

## Notation

- \(P = (x_P, y_P)\): base point (public per constraint instance)
- \(k \in \mathrm{Fr}\): scalar (public per constraint instance)
- Bits \(b_0,\dots,b_{255} \in \{0,1\}\): **MSB-first** bit decomposition of `k`
- Accumulator \(A_i\) (start of step \(i\)), with \(A_0 = \mathcal{O}\)
- Doubled point \(T_i = [2]A_i\)
- Next accumulator \(A_{i+1} = T_i + b_i \cdot P\)

## Witness polynomials (conceptual 8-var MLEs)

Conceptually we have multilinear extensions over \(\{0,1\}^8\) (256 steps):

| Polynomial | Meaning at step \(i\) |
|---|---|
| `x_A(i)`, `y_A(i)` | affine coords of \(A_i\) (or `(0,0)` if \(A_i=\mathcal{O}\)) |
| `x_T(i)`, `y_T(i)` | affine coords of \(T_i=[2]A_i\) (or `(0,0)` if \(T_i=\mathcal{O}\)) |
| `x_A_next(i)`, `y_A_next(i)` | affine coords of \(A_{i+1}\) (shifted “next” values) |
| `ind_A(i)` | indicator for \(A_i=\mathcal{O}\) (intended: 1 if infinity else 0) |
| `ind_T(i)` | indicator for \(T_i=\mathcal{O}\) (intended: 1 if infinity else 0) |

Implementation detail: these 8-var tables are zero-padded to an 11-var table (`2048` entries) to
fit the uniform Dory matrix layout (see `DoryMatrixBuilder::pad_8var_to_11var_zero_padding`).

## Public bit polynomial

The bit values \(b_i\) are **not** committed: the verifier recomputes \(b(r^\*)\) directly from the
public scalar (see `G1ScalarMulPublicInputs::evaluate_bit_mle` in `g1_scalar_mul.rs`).

## Constraint set (matches implementation)

All constraints are equations in `Fq` evaluated at every step index \(i\).

### C1: Doubling x-coordinate (affine, denominator-free)

For \(T = [2]A\) on \(y^2=x^3+3\), using \(\lambda = \frac{3x_A^2}{2y_A}\) and \(x_T=\lambda^2-2x_A\),
we eliminate denominators:

```
C1: 4 y_A(i)^2 * (x_T(i) + 2 x_A(i)) - 9 x_A(i)^4 = 0
```

### C2: Doubling y-coordinate (affine, denominator-free)

Using \(y_T = \lambda(x_A-x_T)-y_A\), we eliminate denominators:

```
C2: 3 x_A(i)^2 * (x_T(i) - x_A(i)) + 2 y_A(i) * (y_T(i) + y_A(i)) = 0
```

### C3: Conditional addition x-coordinate (bit-dependent, with T = O case)

Let \(R = A_{i+1}\), \(T=T_i\). The implemented constraint is:

```
C3: (1 - b(i)) * (x_A_next(i) - x_T(i))
  + b(i) * ind_T(i) * (x_A_next(i) - x_P)
  + b(i) * (1 - ind_T(i)) * [
        (x_A_next(i) + x_T(i) + x_P) * (x_P - x_T(i))^2
      - (y_P - y_T(i))^2
    ] = 0
```

Intended behavior:

- If `b(i)=0`: forces \(A_{i+1}=T_i\) (copy-through).
- If `b(i)=1` and `ind_T(i)=1`: forces \(A_{i+1}=P\) (since \(\mathcal{O}+P=P\)).
- If `b(i)=1` and `ind_T(i)=0`: enforces the standard chord-addition x relation for \(T_i+P\).

### C4: Conditional addition y-coordinate (bit-dependent, with T = O case)

```
C4: (1 - b(i)) * (y_A_next(i) - y_T(i))
  + b(i) * ind_T(i) * (y_A_next(i) - y_P)
  + b(i) * (1 - ind_T(i)) * [
        (y_A_next(i) + y_T(i)) * (x_P - x_T(i))
      - (y_P - y_T(i)) * (x_T(i) - x_A_next(i))
    ] = 0
```

### C6: Doubling preserves infinity (A = O ⇒ T = O)

```
C6: ind_A(i) * (1 - ind_T(i)) = 0
```

### C7: If ind_T = 1 then T has the infinity encoding (0,0)

The implementation enforces:

```
C7: ind_T(i) * (x_T(i)^2 + y_T(i)^2) = 0
```

#### Why this implies x_T = y_T = 0 in BN254 Fq

This implication is **not true over arbitrary fields**, but it **is true** over BN254’s base field
because \(-1\) is a quadratic non-residue in `Fq` (equivalently, `Fq::MODULUS ≡ 3 (mod 4)`).

Proof: if \(x^2+y^2=0\) and \(y\neq 0\), then \((x/y)^2 = -1\), contradiction. Thus \(y=0\), then
\(x=0\).

## What this constraint set does *not* enforce (soundness caveats)

The constraints above are the ones currently implemented. As written, they **do not** justify the
previous “complete soundness / unique trace” claim without extra assumptions or extra constraints.

In particular, the implementation does **not** currently enforce:

- **Indicator booleanity**: `ind_A(i), ind_T(i) ∈ {0,1}`.
- **Curve membership**: when an indicator is 0, the corresponding affine coordinates satisfy
  \(y^2=x^3+3\).
- **Shift consistency across steps**: that `x_A_next(i) = x_A(i+1)` (and similarly for `y` and the
  infinity indicator). Without this, you cannot inductively chain the per-step transition into a
  single length-256 computation trace.
- **Exceptional-case addition** when \(x_T(i)=x_P\) (i.e., \(T_i=\pm P\)). The chord constraints
  are denominator-free but become non-binding when \(T_i=P\) (and they reject the \(T_i=-P\) case,
  where the true sum is \(\mathcal{O}\)).

If you want a spec that truly proves “[k]P” end-to-end against a malicious prover, you must add
constraints (or additional sumchecks) to cover these gaps.

## Hardening sketch for “complete soundness” (not currently implemented)

One reasonable hardening plan (still low-degree) is:

1. **Booleanity for indicators**:

```
ind_A(i) * (1 - ind_A(i)) = 0
ind_T(i) * (1 - ind_T(i)) = 0
```

2. **On-curve when not infinity**:

```
(1 - ind_A(i)) * (y_A(i)^2 - x_A(i)^3 - 3) = 0
(1 - ind_T(i)) * (y_T(i)^2 - x_T(i)^3 - 3) = 0
```

3. **Shift consistency** (needs an additional “shift” argument; see `EqPlusOnePolynomial` used in
`stage1/shift_rho.rs` for the packed-GT case):

- Prove `x_A_next(i) = x_A(i+1)` and `y_A_next(i) = y_A(i+1)` for all \(i\in[0,254]\).
- Prove `ind_A(i+1) = ind_A_next(i)` if you carry a next-indicator.

4. **Handle \(T_i=\pm P\)** either by:

- adding an inverse witness for \((x_P-x_T(i))^{-1}\) when `b(i)=1` and `ind_T(i)=0`, or
- switching to a complete addition law (projective complete formulas) so no inverses/side cases are
  required.

With these additions, an induction-based soundness proof (unique double-and-add trace and correct
final result) becomes valid.
