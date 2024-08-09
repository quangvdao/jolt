# Specification of Jolt Subtables

## Notation

- $m$ is the number of bits for each operand.

- $x$ is a $m$-bit natural number, expressed in binary as $(x_{m-1}, x_{m-2}, \ldots, x_1, x_0) \in \{0, 1\}^m$.

- $y$ is a $m$-bit natural number, expressed in binary as $(y_{m-1}, y_{m-2}, \ldots, y_1, y_0) \in \{0, 1\}^m$.

- Sometimes, we use $z = x \| y$ to denote the concatenation of $x$ and $y$, indexed as $z = (z_{2m-1}, z_{2m-2}, \ldots, z_1, z_0) \in \{0, 1\}^{2m}$.

- We define subrange as $[a,b] := \{a, a+1, \ldots, b\}$.

- The subtable value is a natural number having at most $2m$ bits, embedded in a larger prime field.

- Jolt internally represents subtables as a single vector of $M:=2^{2m}$ field elements. The indexing is done in the natural way, by interpreting the concatenation $z := x \| y$ as a $2m$-bit natural number.

## Subtables List

1. `AndSubtable`

This subtable stores the bitwise AND of the inputs, interpreted as a $m$-bit natural number.

Function: $ And(x, y) = (x_1 \land y_1, x_2 \land y_2, \ldots, x_m \land y_m) $

Multilinear extension: $\widetilde{And}(x, y) = \prod_{i=0}^{m-1} 2^{m - i - 1} \cdot x_i \cdot y_i $

2. `DivByZeroSubtable`

This subtable stores `1` if the inputs are $x=0$ and $y=2^m - 1$, and `0` otherwise.

Function: $ DivByZero(x, y) = \begin{cases} 1 & \text{if } (x_i,y_i)=(0,1) \text{ for all } i \\ 0 & \text{otherwise} \end{cases} $

Multilinear extension: $\widetilde{DivByZero}(x, y) = \prod_{i=0}^{m-1} \left( (1 - x_i) \cdot y_i \right) $

3. `EqAbsSubtable`

This subtable stores `1` if the inputs are equal ignoring the first bit, and `0` otherwise.

Function: $ EqAbs(x, y) = \begin{cases} 1 & \text{if } x[1,m-1] = y[1,m-1] \\ 0 & \text{if } x \neq y \end{cases} $

Multilinear extension: $\widetilde{EqAbs}(x, y) = \prod_{i=1}^{m-1} \left( (1 - x_i) \cdot (1 - y_i) + x_i \cdot y_i \right) $

4. `EqSubtable`

This subtable stores `1` if the inputs are equal, and `0` otherwise.

Function: $ Eq(x, y) = \begin{cases} 1 & \text{if } x = y \\ 0 & \text{if } x \neq y \end{cases} $

Multilinear extension: $\widetilde{Eq}(x, y) = \prod_{i=0}^{m-1} \left( (1 - x_i) \cdot (1 - y_i) + x_i \cdot y_i \right) $

5. `IdentitySubtable`

This subtable stores the result of two inputs concatenated together, interpreted as a $2m$-bit number.

Function: $ Id(z) = z $

Multilinear extension: $\widetilde{Id}(z) = \sum_{i=0}^{m-1} 2^{m - i - 1} \cdot z_{i} $

6. `LeftIsZeroSubtable`

This subtable stores `1` if and only if the first input is zero.

Function: $ LeftIsZero(x, y) = \begin{cases} 1 & \text{if } x = 0 \\ 0 & \text{if } x \neq 0 \end{cases} $

Multilinear extension: $\widetilde{LeftIsZero}(x, y) = \prod_{i=0}^{m-1} (1 - x_i) $

7. `LeftMSBSubtable`

This subtable stores the most significant bit of the first input.

Function: $ LeftMSB(x, y) = x[0] $

Multilinear extension: $\widetilde{LeftMSB}(x, y) = x[0] $

8. `LtAbsSubtable`

This subtable stores `1` if the first input is less than the second input, after ignoring the first bit, and `0` otherwise.

Function: $ LtAbs(x, y) = \begin{cases} 1 & \text{if } x[1,m-1] < y[1,m-1] \\ 0 & \text{otherwise} \end{cases} $

Multilinear extension: $\widetilde{LtAbs}(x, y) = \sum_{i=1}^{m-1} (1 - x_i) \cdot y_i \cdot \prod_{j=0}^{i-1} ((1 - x_j) \cdot (1 - y_j) + x_j \cdot y_j) $

9. `LtuSubtable`

This subtable stores `1` if the first input is less than the second input, and `0` otherwise.

Function: $ Ltu(x, y) = \begin{cases} 1 & \text{if } x < y \\ 0 & \text{otherwise} \end{cases} $

Multilinear extension: $\widetilde{Ltu}(x, y) = \sum_{i=0}^{m-1} (1 - x_i) \cdot y_i \cdot \prod_{j=0}^{i-1} ((1 - x_j) \cdot (1 - y_j) + x_j \cdot y_j) $

10. `OrSubtable`

This subtable stores the bitwise OR of the inputs, interpreted as a $m$-bit natural number.

Function: $ Or(x, y) = (x_0 \lor y_0, x_1 \lor y_1, \ldots, x_{m-1} \lor y_{m-1}) $

Multilinear extension: $\widetilde{Or}(x, y) = \sum_{i=0}^{m-1} 2^{m - i - 1} \cdot \left( 2^m \cdot (x_i + y_i - x_i \cdot y_i) \right) $

11. `RightIsZeroSubtable`

This subtable stores `1` if and only if the second input is zero.

Function: $ RightIsZero(x, y) = \begin{cases} 1 & \text{if } y = 0 \\ 0 & \text{if } y \neq 0 \end{cases} $

Multilinear extension: $\widetilde{RightIsZero}(x, y) = \prod_{i=0}^{m-1} (1 - y_i) $

12. `RightMSBSubtable`

This subtable stores the most significant bit of the second input.

Function: $ RightMSB(x, y) = y[0] $

Multilinear extension: $\widetilde{RightMSB}(x, y) = y[0] $

13. `SignExtendSubtable`

(TODO: double-check this description)

Has a `WIDTH` parameter, which is the width of the value being sign-extended.

This subtable stores the sign-extended value of the first `WIDTH` bits of the input, interpreted as a `WIDTH`-bit number and extended to $2m$ bits. The subtable is designed to work with $M = 2^{16}$.

Function: For an input $x$, let $s = x[W-1]$ be the sign bit, where $W$ is the width. Then:

$SignExtend(x) = \begin{cases} 
x[0, W-1] \| 0^{2m-W} & \text{if } s = 0 \\
x[0, W-1] \| 1^{2m-W} & \text{if } s = 1
\end{cases}$

Multilinear extension: $\widetilde{SignExtend}(x) = s \cdot (2^W - 1)$

where $s$ is the $(W-1)$-th bit of the input.

Note: The current implementation assumes $M = 2^{16}$ and extends the sign to fill the entire $WIDTH$-bit output, rather than extending to $2m$ bits. This is designed for specific use cases in LB and LH operations.


14. `SllSubtable`

This subtable implements a logical left shift operation on a chunk of bits. It has three type parameters:
- `F`: The field type
- `CHUNK_INDEX`: The index of the chunk being shifted
- `WORD_SIZE`: The total size of the word in bits

Function: For inputs $x$ and $y$, where $x$ is the chunk to be shifted and $y$ is the shift amount:

$Sll(x, y) = (x \ll (y \bmod WORD\_SIZE + suffix\_length)) \gg suffix\_length$

where $suffix\_length = operand\_chunk\_width * CHUNK\_INDEX$, and $operand\_chunk\_width = \log_2(M) / 2$.

The operation performs the following steps:
1. Left-shift $x$ by $(y \bmod WORD\_SIZE + suffix\_length)$
2. Take the result modulo $2^{WORD\_SIZE}$ to wrap around
3. Right-shift by $suffix\_length$ to align the result

Multilinear extension: The MLE is more complex and involves summing over possible shift amounts:

$\widetilde{Sll}(x, y) = \sum_{k=0}^{\min(WORD\_SIZE-1, 2^b-1)} eq\_term_k \cdot shift\_x\_by\_k$

where:
- $b$ is half the number of input bits
- $eq\_term_k = \prod_{i=0}^{\min(\log_2(WORD\_SIZE)-1, b-1)} (k_i \cdot y_i + (1 - k_i) \cdot (1 - y_i))$
- $k_i$ is the $i$-th bit of $k$ in big-endian order
- $shift\_x\_by\_k = \sum_{j=0}^{m'-1} 2^{j+k} \cdot x_{b-1-j}$
- $m = \min(b, (k + b \cdot (CHUNK\_INDEX + 1)) - WORD\_SIZE)$ if $(k + b \cdot (CHUNK\_INDEX + 1)) > WORD\_SIZE$, else 0
- $m' = b - m$

This subtable handles the complexities of shifting within a specific chunk of a larger word, taking into account potential overflow and alignment issues.

15. `SraSignSubtable`

(TODO: double-check this description)

This subtable implements the sign extension part of an arithmetic right shift operation. It has two type parameters:
- `F`: The field type
- `WORD_SIZE`: The total size of the word in bits

Function: For inputs $x$ and $y$, where $x$ is the chunk containing the sign bit and $y$ is the shift amount:

$SraSign(x, y) = \sum_{i=0}^{(y \bmod WORD\_SIZE) - 1} 2^{WORD\_SIZE - 1 - i} \cdot sign\_bit$

where $sign\_bit = x_{sign\_bit\_index}$ and $sign\_bit\_index = (WORD\_SIZE - 1) \bmod operand\_chunk\_width$.

The $operand\_chunk\_width$ is calculated as $\log_2(M) / 2$, where $M$ is the size of the subtable.

Multilinear extension:

$\widetilde{SraSign}(x, y) = \sum_{k=0}^{\min(WORD\_SIZE-1, 2^b-1)} eq\_term_k \cdot x\_sign\_upper_k$

where:
- $b$ is half the number of input bits
- $eq\_term_k = \prod_{i=0}^{\min(\log_2(WORD\_SIZE)-1, b-1)} (k_i \cdot y_i + (1 - k_i) \cdot (1 - y_i))$
- $k_i$ is the $i$-th bit of $k$ in big-endian order
- $x\_sign\_upper_k = \sum_{i=0}^{k-1} 2^{WORD\_SIZE - 1 - i} \cdot x_{sign}$
- $x_{sign}$ is the sign bit of $x$, located at index $(WORD\_SIZE - 1) \bmod b$ in the $x$ input vector

This subtable handles the complexities of sign extension during an arithmetic right shift, taking into account the position of the sign bit within the input chunk and the variable shift amount.


16. `SrlSubtable`

(TODO: double-check this description)

This subtable implements a logical right shift operation on a chunk of bits. It has three type parameters:
- `F`: The field type
- `CHUNK_INDEX`: The index of the chunk being shifted
- `WORD_SIZE`: The total size of the word in bits

Function: For inputs $x$ and $y$, where $x$ is the chunk to be shifted and $y$ is the shift amount:

$Srl(x, y) = (x \ll suffix\_length) \gg (y \bmod WORD\_SIZE)$

where $suffix\_length = operand\_chunk\_width * CHUNK\_INDEX$, and $operand\_chunk\_width = \log_2(M) / 2$.

The operation performs the following steps:
1. Left-shift $x$ by $suffix\_length$ to align it within the word
2. Right-shift the result by $(y \bmod WORD\_SIZE)$

Multilinear extension: The MLE involves summing over possible shift amounts:

$\widetilde{Srl}(x, y) = \sum_{k=0}^{\min(WORD\_SIZE-1, 2^b-1)} eq\_term_k \cdot shift\_x\_by\_k$

where:
- $b$ is half the number of input bits
- $eq\_term_k = \prod_{i=0}^{\min(\log_2(WORD\_SIZE)-1, b-1)} (k_i \cdot y_i + (1 - k_i) \cdot (1 - y_i))$
- $k_i$ is the $i$-th bit of $k$ in big-endian order
- $m = \min(b, k - b \cdot CHUNK\_INDEX)$ if $k > b \cdot CHUNK\_INDEX$, else 0
- $chunk\_length = b - \max(0, (b \cdot (CHUNK\_INDEX + 1)) - WORD\_SIZE)$
- $shift\_x\_by\_k = \sum_{j=m}^{chunk\_length-1} 2^{b \cdot CHUNK\_INDEX + j - k} \cdot x_{b-1-j}$

This subtable handles the complexities of shifting within a specific chunk of a larger word, taking into account potential underflow and alignment issues.


17. `TruncateOverflowSubtable`

(TODO: double-check this description)

This subtable truncates the input to the least significant bits that fit within the specified `WORD_SIZE`, effectively handling overflow by discarding the most significant bits.

Function: For an input $x$, let $c = WORD\_SIZE \bmod \log_2(M)$. Then:

$TruncateOverflow(x) = x \bmod 2^c$

Multilinear extension: $\widetilde{TruncateOverflow}(x) = \sum_{i=0}^{c-1} 2^i \cdot x_{m - i - 1}$

where $c = WORD\_SIZE \bmod \log_2(M)$, and $x_i$ represents the $i$-th bit of the input $x$ (0-indexed from least significant).

Note: This subtable is parameterized by `WORD_SIZE` and adapts to the size of the field $M$. It retains only the least significant $c$ bits of the input, where $c$ is the remainder when `WORD_SIZE` is divided by $\log_2(M)$.


18. `XorSubtable`

This subtable stores the bitwise XOR of the inputs, interpreted as a $m$-bit natural number.

Function: $ Xor(x, y) = (x_0 \oplus y_0, x_1 \oplus y_1, \ldots, x_{m-1} \oplus y_{m-1}) $

Multilinear extension: $\widetilde{Xor}(x, y) = \sum_{i=0}^{m-1} 2^{m - i - 1} \cdot ((1 - x_i) \cdot y_i + x_i \cdot (1 - y_i)) $


19. `ZeroLSBSubtable`

This subtable stores the input value with its least significant bit (LSB) set to zero, interpreted as an $m$-bit natural number.

Function: $ZeroLSB(z) = z - (z \% 2)$

Multilinear extension: $\widetilde{ZeroLSB}(z) = \sum_{i=1}^{m-1} 2^{m - i} \cdot z_{i}$

Note: This subtable effectively clears the LSB of the input, preserving all other bits.