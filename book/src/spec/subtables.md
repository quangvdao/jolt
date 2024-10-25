# Specification of Jolt Subtables

A Jolt subtable is a lookup table consisting of $2^{2m}$ field elements (indexed by a tuple of two $m$-bit natural numbers $(x,y)$), whose multilinear extension (MLE) has a simple algebraic form.

In this page, we give a full specification of all Jolt subtables for the current RV32IM version over curves. The subtables may change in future versions of Jolt.

In the codebase, each subtable must implement the methods described below.

```rust
pub trait LassoSubtable<F: JoltField>: 'static + Sync {

    fn subtable_id(&self) -> SubtableId { TypeId::of::<Self>() }
    
    fn materialize(&self, M: usize) -> Vec<F>;

    fn evaluate_mle(&self, point: &[F]) -> F;
}
```

- `subtable_id` returns the unique identifier for the subtable.

- `materialize` returns a vector of size `M` containing the subtable values.

- `evaluate_mle` returns the value of the multilinear extension polynomial for the subtable at the given `point`, interpreted to be of size `log_2(M)`, where `M` is the size of the subtable.

## Notation

- `M` is the subtable size, currently set to be `2 ^ 16` for all subtables.

- `m = log_2(M) / 2 = 8` is the bit-length of each half of the operand going into the subtable.

- `WORD_SIZE`: The total size of the word in bits (either 32 or 64)

- `CHUNK_INDEX`: For shift instructions, this is the index of the chunk being shifted

- $x$ is a $m$-bit natural number, expressed in binary as $(x_{m-1}, x_{m-2}, \ldots, x_1, x_0) \in \{0, 1\}^m$.

- $y$ is a $m$-bit natural number, expressed in binary as $(y_{m-1}, y_{m-2}, \ldots, y_1, y_0) \in \{0, 1\}^m$.

- Sometimes, we use $z = x \| y$ to denote the concatenation of $x$ and $y$, indexed as $z = (z_{2m-1}, z_{2m-2}, \ldots, z_1, z_0) \in \{0, 1\}^{2m}$.

- We define subrange as $[a,b] := \{a, a+1, \ldots, b\}$.

- The subtable value is a natural number having at most $2m$ bits, embedded in a larger prime field.

- Jolt internally represents subtables as a single vector of $M:=2^{2m}$ field elements. The indexing is done in the natural way, by interpreting the concatenation $z := x \| y$ as a $2m$-bit natural number.

## Subtables List

<!-- markdownlint-disable MD029 -->

1. [`AndSubtable`](../../../jolt-core/src/jolt/subtable/and.rs)

This subtable stores the bitwise AND of the inputs, interpreted as a $m$-bit natural number.

- **Function:** $\mathsf{And}_m(x, y) = (x_1 \land y_1, x_2 \land y_2, \ldots, x_m \land y_m) $

- **Multilinear extension:** $\widetilde{\mathsf{And}}_m(x, y) = \prod_{i=0}^{m-1} 2^{m - i - 1} \cdot x_i \cdot y_i $

2. [`DivByZeroSubtable`](../../../jolt-core/src/jolt/subtable/div_by_zero.rs)

This subtable stores `1` if the inputs are $x=0$ and $y=2^m - 1$, and `0` otherwise.

- **Function:** $\mathsf{DivByZero}_m(x, y) = \begin{cases} 1 & \text{if } (x_i,y_i)=(0,1) \text{ for all } i \\ 0 & \text{otherwise} \end{cases} $

- **Multilinear extension:** $\widetilde{\mathsf{DivByZero}}_m(x, y) = \prod_{i=0}^{m-1} \left( (1 - x_i) \cdot y_i \right) $

3. [`EqAbsSubtable`](../../../jolt-core/src/jolt/subtable/eq_abs.rs)

This subtable stores `1` if the inputs are equal ignoring the first bit, and `0` otherwise.

- **Function:** $\mathsf{EqAbs}_m(x, y) = \begin{cases} 1 & \text{if } x[1,m-1] = y[1,m-1] \\ 0 & \text{if } x \neq y \end{cases} $

- **Multilinear extension:** $\widetilde{\mathsf{EqAbs}}_m(x, y) = \prod_{i=1}^{m-1} \left( (1 - x_i) \cdot (1 - y_i) + x_i \cdot y_i \right) $

4. [`EqSubtable`](../../../jolt-core/src/jolt/subtable/eq.rs)

This subtable stores `1` if the inputs are equal, and `0` otherwise.

- **Function:** $\mathsf{Eq}_m(x, y) = \begin{cases} 1 & \text{if } x = y \\ 0 & \text{if } x \neq y \end{cases} $

- **Multilinear extension:** $\widetilde{\mathsf{Eq}}_m(x, y) = \prod_{i=0}^{m-1} \left( (1 - x_i) \cdot (1 - y_i) + x_i \cdot y_i \right) $

5. [`IdentitySubtable`](../../../jolt-core/src/jolt/subtable/identity.rs)

This subtable stores the result of two inputs concatenated together, interpreted as a $2m$-bit number.

- **Function:** $\mathsf{Id}_m(z) = z $

- **Multilinear extension:** $\widetilde{\mathsf{Id}}_m(z) = \sum_{i=0}^{m-1} 2^{m - i - 1} \cdot z_{i} $

6. [`LeftIsZeroSubtable`](../../../jolt-core/src/jolt/subtable/left_is_zero.rs)

This subtable stores `1` if and only if the first input is zero.

- **Function:** $\mathsf{LeftIsZero}_m(x, y) = \begin{cases} 1 & \text{if } x = 0 \\ 0 & \text{if } x \neq 0 \end{cases} $

- **Multilinear extension:** $\widetilde{\mathsf{LeftIsZero}}_m(x, y) = \prod_{i=0}^{m-1} (1 - x_i) $

7. [`LeftMSBSubtable`](../../../jolt-core/src/jolt/subtable/left_msb.rs)

This subtable stores the most significant bit of the first input.

- **Function:** $\mathsf{LeftMSB}_m(x, y) = x[0] $

- **Multilinear extension:** $\widetilde{\mathsf{LeftMSB}}_m(x, y) = x[0] $

8. [`LtAbsSubtable`](../../../jolt-core/src/jolt/subtable/lt_abs.rs)

This subtable stores `1` if the first input is less than the second input, after ignoring the first bit, and `0` otherwise.

- **Function:** $\mathsf{LtAbs}_m(x, y) = \begin{cases} 1 & \text{if } x[1,m-1] < y[1,m-1] \\ 0 & \text{otherwise} \end{cases} $

- **Multilinear extension:** $\widetilde{\mathsf{LtAbs}}_m(x, y) = \sum_{i=1}^{m-1} (1 - x_i) \cdot y_i \cdot \prod_{j=0}^{i-1} ((1 - x_j) \cdot (1 - y_j) + x_j \cdot y_j) $

9. [`LtuSubtable`](../../../jolt-core/src/jolt/subtable/ltu.rs)

This subtable stores `1` if the first input is less than the second input, and `0` otherwise.

- **Function:** $\mathsf{Ltu}_m(x, y) = \begin{cases} 1 & \text{if } x < y \\ 0 & \text{otherwise} \end{cases} $

- **Multilinear extension:** $\widetilde{\mathsf{Ltu}}_m(x, y) = \sum_{i=0}^{m-1} (1 - x_i) \cdot y_i \cdot \prod_{j=0}^{i-1} ((1 - x_j) \cdot (1 - y_j) + x_j \cdot y_j) $

10. [`OrSubtable`](../../../jolt-core/src/jolt/subtable/or.rs)

This subtable stores the bitwise OR of the inputs, interpreted as a $m$-bit natural number.

- **Function:** $\mathsf{Or}_m(x, y) = (x_0 \lor y_0, x_1 \lor y_1, \ldots, x_{m-1} \lor y_{m-1}) $

- **Multilinear extension:** $\widetilde{\mathsf{Or}}_m(x, y) = \sum_{i=0}^{m-1} 2^{m - i - 1} \cdot \left( 2^m \cdot (x_i + y_i - x_i \cdot y_i) \right) $

11. [`RightIsZeroSubtable`](../../../jolt-core/src/jolt/subtable/right_is_zero.rs)

This subtable stores `1` if and only if the second input is zero.

- **Function:** $\mathsf{RightIsZero}_m(x, y) = \begin{cases} 1 & \text{if } y = 0 \\ 0 & \text{if } y \neq 0 \end{cases} $

- **Multilinear extension:** $\widetilde{\mathsf{RightIsZero}}_m(x, y) = \prod_{i=0}^{m-1} (1 - y_i) $

12. [`RightMSBSubtable`](../../../jolt-core/src/jolt/subtable/right_msb.rs)

This subtable stores the most significant bit of the second input.

- **Function:** $\mathsf{RightMSB}_m(x, y) = y[0] $

- **Multilinear extension:** $\widetilde{\mathsf{RightMSB}}_m(x, y) = y[0] $

13. [`SignExtendSubtable`](../../../jolt-core/src/jolt/subtable/sign_extend.rs)

This subtable is further parametrized by a width $w \le 2m$ (denoted in Jolt as `WIDTH`). Informally, this subtable stores either all-zero or all-one (for $w$ bits) depending on the $w$-th least significant bit of $z = x \| y$ (interpreted as a sign bit).

- **Function:** $\mathsf{SignExtend}_{m,w}(z) = \begin{cases} 0 & \text{if } z_{2m - w} = 0 \\
        2^w - 1 & \text{if } z_{2m - w} = 1 \end{cases}$

- **Multilinear extension:** $\widetilde{\mathsf{SignExtend}}_{m,w}(Z) = (2^w - 1) \cdot Z_{2m - w}$

- *Implementation Note:* The current implementation assumes $M = 2^{16}$ and extends the sign to fill the entire `WIDTH`-bit output, rather than extending to $2m$ bits.

14. [`SllSubtable`](../../../jolt-core/src/jolt/subtable/sll.rs)

This subtable is further parametrized by two values, the chunk index $i$ (or `CHUNK_INDEX`) and the word size $W$ (or `WORD_SIZE`). Informally, this subtable assumes that $x$ is the $i$-th chunk of some $W$-bit number, shifts it left by $y \bmod W$ bits, then truncates the result to be $W - m \cdot i$ bits. Recall that each chunk of a $W$-sized number $x$ has $m$ bits, and the indexing goes from left-to-right, so the first chunk is the most significant bits of $x$.

- **Function:** $\mathsf{Sll}_{m,i,W}(x, y) = (x \ll (y \bmod W)) \bmod 2^{W - m \cdot i} $

- **Multilinear extension:** $\widetilde{\mathsf{Sll}}_{m,i,W}(X, Y) = \sum_{k=0}^{2^{m'}-1} \widetilde{\mathsf{Eq}}_{m'}(Y,k) \cdot \left(\sum_{j=0}^{m - n_{\mathsf{out}}-1} 2^{k + j} \cdot X_{m - j - 1}\right) $

15. [`SraSignSubtable`](../../../jolt-core/src/jolt/subtable/sra_sign.rs)

This subtable is further parametrized by the word size $W$ (`WORD_SIZE` in code), and stores $x_{(m - 1) - ((W-1) \bmod m)}$ (interpreted as a sign bit) duplicated $y \bmod W$ times in the most significant bits of the result (which is a $W$-bit number). In other words, if $x$'s most significant bit is `0`, then the result is `0`. Otherwise, the result is the number `11..10..0` with the same number of `1`s as the shift amount, which is $y \bmod W$.

- **Function:** $\mathsf{SraSign}_{m,W}(x, y) = \begin{cases} 0 & \text{if } x_{i_{\mathsf{sign}}} = 0 \\
        \sum_{i=0}^{(y \bmod W) - 1} 2^{W - 1 - i} & \text{if } x_{i_{\mathsf{sign}}} = 1 \end{cases}$

- **Multilinear extension:** Let $m' = \min(m, \lceil \log_2(W) \rceil)$. Then:
    $$\widetilde{\mathsf{SraSign}}_{m,W}(X, Y) = \sum_{k=0}^{2^{m'} - 1} \widetilde{\mathsf{Eq}}_{m'}(Y,k) \cdot X_{i_{\mathsf{sign}}} \cdot \left(\sum_{j=0}^{k - 1} 2^{W - 1 - j}\right).$$

16. [`SrlSubtable`](../../../jolt-core/src/jolt/subtable/srl.rs)

This subtable is further parametrized by two values, the chunk index $i$ (`CHUNK_INDEX` in code) and the word size $W$ (`WORD_SIZE` in code). Informally, this subtable assumes that $x$ is the $i$-th chunk of some $W$-bit number, shifts it left by $m \cdot i$ bits to align with the pre-chunk position, then shifts it right by $y \bmod W$ bits.

- **Function:** $\mathsf{Srl}_{m,i,W}(x, y) = (x \ll m \cdot i) \gg (y \bmod W)$

- **Multilinear extension:** Let $m' = \min(m, \lceil \log_2(W) \rceil)$, $m''=\min(m,W - m \cdot i)$, and $n_{\mathsf{out}} = \min\left(m, \max \left(0, k + m \cdot (i + 1) - W\right)\right)$ denotes the number of bits that goes out of range. Then:
    $$\widetilde{\mathsf{Srl}}_{m,i,W}(X, Y) = \sum_{k=0}^{2^{m'}-1} \widetilde{\mathsf{Eq}}_{m'}(Y,k) \cdot \left(\sum_{j=n_{\mathsf{out}}}^{m'' - 1} 2^{m \cdot (i - 1) - k + j} \cdot X_{m - j - 1}\right).$$

17. [`TruncateOverflowSubtable`](../../../jolt-core/src/jolt/subtable/truncate_overflow.rs)

This subtable is further parametrized by a word size $W$. Informally, this subtable truncates $z = x \| y$ to $w$ bits and then zero-extends it to $2m$ bits, where $w = W \bmod (2m)$ is the number of overflow bits.

- **Function:** $\mathsf{TruncateOverflow}_{m,W}(z) = z \bmod 2^{W \bmod (2m)}$

- **Multilinear extension:** $\widetilde{\mathsf{TruncateOverflow}}_{m,W}(Z) = \sum_{i=0}^{(W \bmod 2m) - 1} 2^i \cdot Z_{m - i - 1}$

18. [`XorSubtable`](../../../jolt-core/src/jolt/subtable/xor.rs)

This subtable stores the bitwise XOR of the inputs, interpreted as a $m$-bit natural number.

- **Function:** $\mathsf{Xor}_m(x, y) = (x_0 \oplus y_0, x_1 \oplus y_1, \ldots, x_{m-1} \oplus y_{m-1}) $

- **Multilinear extension:** $\widetilde{\mathsf{Xor}}_m(x, y) = \sum_{i=0}^{m-1} 2^{m - i - 1} \cdot ((1 - x_i) \cdot y_i + x_i \cdot (1 - y_i)) $

19. [`ZeroLSBSubtable`](../../../jolt-core/src/jolt/subtable/zero_lsb.rs)

This subtable stores the input value with its least significant bit (LSB) set to zero, and all other bits remaining the same.

- **Function:** $\mathsf{ZeroLSB}_m(z) = z - (z \% 2)$

- **Multilinear extension:** $\widetilde{\mathsf{ZeroLSB}}_m(z) = \sum_{i=1}^{m-1} 2^{m - i} \cdot z_{i}$
