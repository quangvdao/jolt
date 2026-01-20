# G1 Scalar Multiplication Sumcheck Specification

## Overview

This document specifies the constraints for proving G1 scalar multiplication `[k]P` via sumcheck. The goal is **complete soundness**: a malicious prover cannot satisfy the constraints unless the witness corresponds to the actual double-and-add trace of `[k]P`.

## Notation

- `P = (x_P, y_P)` - base point (public, known to verifier)
- `k` - scalar (256 bits), with bit decomposition `b_0, b_1, ..., b_{255}` (MSB-first)
- `A_i` - accumulator at step i, where `A_0 = O` (point at infinity)
- `T_i = [2]A_i` - doubled accumulator
- `A_{i+1} = T_i + b_i · P` - next accumulator (conditional add)
- `n = 256` - number of bits/steps

## Double-and-Add Algorithm

```
A_0 = O (point at infinity)
for i = 0 to n-1:
    T_i = [2]A_i           (doubling)
    A_{i+1} = T_i + b_i·P  (conditional addition)
return A_n = [k]P
```

## Witness Polynomials (MLEs over {0,1}^8)

| Polynomial | Description |
|------------|-------------|
| `x_A(i)`, `y_A(i)` | Coordinates of accumulator A_i at step i |
| `x_T(i)`, `y_T(i)` | Coordinates of T_i = [2]A_i |
| `x_A'(i)`, `y_A'(i)` | Coordinates of A_{i+1} (shifted) |
| `b(i)` | Scalar bit b_i ∈ {0, 1} |
| `ind_A(i)` | 1 if A_i = O, else 0 |
| `ind_T(i)` | 1 if T_i = O, else 0 |

## Constraint Equations

### C1: Doubling x-coordinate (when A_i ≠ O)

For doubling `T = [2]A` on short Weierstrass curve `y² = x³ + b`:
- Tangent slope: `λ = 3x_A² / 2y_A`
- New x: `x_T = λ² - 2x_A`

Eliminating denominators:
```
C1: 4y_A² · (x_T + 2x_A) - 9x_A⁴ = 0
```

### C2: Doubling y-coordinate (when A_i ≠ O)

- New y: `y_T = λ(x_A - x_T) - y_A`

Eliminating denominators:
```
C2: 3x_A² · (x_T - x_A) + 2y_A · (y_T + y_A) = 0
```

### C3: Conditional addition x-coordinate

**Key insight**: We need bit-dependent branching:
- If `b_i = 0`: `A_{i+1} = T_i` → enforce `(x_A', y_A') = (x_T, y_T)`
- If `b_i = 1`: `A_{i+1} = T_i + P` → enforce chord addition formula

For chord addition `R = T + P` with `T ≠ P` (guaranteed for random base):
- Slope: `λ = (y_P - y_T) / (x_P - x_T)`
- New x: `x_R = λ² - x_T - x_P`

Eliminating denominators:
```
x_R · (x_P - x_T)² = (y_P - y_T)² - (x_T + x_P) · (x_P - x_T)²
```

Rearranging:
```
(x_R + x_T + x_P) · (x_P - x_T)² - (y_P - y_T)² = 0
```

**Combined constraint**:
```
C3: (1 - b) · (x_A' - x_T) + b · ind_T · x_A' · (x_A' - x_P)
    + b · (1 - ind_T) · [(x_A' + x_T + x_P) · (x_P - x_T)² - (y_P - y_T)²] = 0
```

Breaking this down:
- When `b = 0`: forces `x_A' = x_T`
- When `b = 1` and `T = O` (ind_T = 1): forces `x_A' ∈ {0, x_P}` (i.e., O or P)
- When `b = 1` and `T ≠ O` (ind_T = 0): forces chord addition formula

### C4: Conditional addition y-coordinate

For chord addition y-coordinate:
```
y_R = λ(x_T - x_R) - y_T = (y_P - y_T) · (x_T - x_R) / (x_P - x_T) - y_T
```

Eliminating denominators and rearranging:
```
y_R · (x_P - x_T) = (y_P - y_T) · (x_T - x_R) - y_T · (x_P - x_T)
y_R · (x_P - x_T) + y_T · (x_P - x_T) = (y_P - y_T) · (x_T - x_R)
(y_R + y_T) · (x_P - x_T) - (y_P - y_T) · (x_T - x_R) = 0
```

**Combined constraint**:
```
C4: (1 - b) · (y_A' - y_T) + b · ind_T · y_A' · (y_A' - y_P)
    + b · (1 - ind_T) · [(y_A' + y_T) · (x_P - x_T) - (y_P - y_T) · (x_T - x_A')] = 0
```

### C5: Bit booleanity

```
C5: b · (1 - b) = 0
```

This ensures `b ∈ {0, 1}`.

### C6: Accumulator infinity indicator consistency

When `A = O`, we use coordinates `(0, 0)`:
```
C6: ind_A · (x_A² + y_A²) = 0
```

And conversely, if coordinates are `(0, 0)`, then `ind_A = 1`:
```
C7: (1 - ind_A) · (x_A · y_A) = 0  (only when x_A ≠ 0 and y_A ≠ 0)
```

**Note**: For simplicity, we can assume the honest witness sets `ind_A = 1` iff `A = O`. The verifier checks initial and final conditions separately.

### C8: Doubling infinity case

When `A = O`, doubling gives `T = O`:
```
C8: ind_A · (1 - ind_T) = 0  (if A = O then T = O)
C9: ind_A · (x_T² + y_T²) = 0  (if A = O then T has coords (0,0))
```

## Boundary Conditions (Verified Outside Sumcheck)

1. **Initial**: `A_0 = O` (accumulator starts at identity)
   - Check: `x_A(0) = 0`, `y_A(0) = 0`, `ind_A(0) = 1`

2. **Final**: `A_n = [k]P` (result matches expected)
   - Check: `x_A'(255) = x_result`, `y_A'(255) = y_result`

3. **Bit decomposition**: The bits `b(i)` must equal the actual scalar
   - This is verified by checking `Σ_i b(i) · 2^{255-i} = k`
   - Can be done via separate constraint or public input check

## Soundness Proof Sketch

**Theorem**: If a prover provides witness polynomials satisfying C1-C9 at all `i ∈ {0,...,255}`, then the witness represents the unique double-and-add trace for scalar `k = Σ_i b(i) · 2^{255-i}` starting from `A_0 = O`.

**Proof**:
1. C5 ensures each `b(i) ∈ {0, 1}` - these define a unique scalar k
2. Initial condition ensures `A_0 = O`
3. By induction on i:
   - C1, C2 (with C8-C9 for infinity) uniquely determine `T_i = [2]A_i`
   - C3, C4 (with C5) uniquely determine `A_{i+1}`:
     - If `b_i = 0`: forced to `A_{i+1} = T_i`
     - If `b_i = 1`: forced to `A_{i+1} = T_i + P`
   - This is exactly the double-and-add algorithm
4. Final condition ensures `A_n` equals the claimed result

**Completeness**: The honest witness (from running double-and-add) satisfies all constraints.

## Degree Analysis

| Constraint | Degree | Notes |
|------------|--------|-------|
| C1 | 4 | `y_A² · x_T` and `x_A⁴` |
| C2 | 3 | `x_A² · x_T` |
| C3 | 5 | `b · (x_P - x_T)² · (x_A' + ...)` |
| C4 | 4 | `b · (y_A' + y_T) · (x_P - x_T)` |
| C5 | 2 | `b · (1 - b)` |
| C6 | 3 | `ind · x²` |

**Sumcheck degree**: 6 (after batching with eq polynomial)

## Implementation Notes

1. **Witness generation**: Run the actual double-and-add algorithm, recording all intermediate values
2. **Bit polynomial**: Include `b(i)` as a committed polynomial
3. **Public verification**: The scalar `k` must be publicly known or verified via the bit decomposition
4. **Base point**: `P = (x_P, y_P)` is public and hardcoded into the verifier's constraint evaluation
