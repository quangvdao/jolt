# Specification of Jolt Instructions

We describe the instructions that Jolt uses, including how they are implemented in the codebase, and their specification in mathematical notation.

In the codebase, each instruction must implement the methods described below.

```rust
pub trait JoltInstruction: Clone + Debug + Send + Sync + Serialize {

    fn operands(&self) -> (u64, u64);

    fn to_indices(&self, C: usize, M: usize) -> Vec<usize>;

    fn subtables<F: JoltField>(&self, C: usize, M: usize) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)>;

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F;

    fn lookup_entry(&self) -> u64;

    // Other methods omitted for brevity
}
```

- `operands` returns the two operands of the instruction. For instructions that take a single operand, the second operand is set to some constant value (specified for each instruction).

- `to_indices` returns the chunking of the operands into $C$ chunks of size $m := \log_2(M)/2$, with the last chunk being shorter if the operand is not a multiple of $m$.

- `subtables` returns a list of subtables (and their indices) that are used to evaluate the instruction. For each tuple in the list, the first element is a `LassoSubtable`, and the second element is a `SubtableIndices`, which specifies which chunks of the operands are used in the evaluation.

- `combine_lookups` takes the values returned by the subtables and combines them to compute the final result of the instruction.

- `lookup_entry` returns the expected output of the instruction.

Note that the RISC-V instructions are not the same as the Jolt instructions. In fact, several RISC-V instructions may invoke the same Jolt instruction (see [rv.rs](../../../jolt-core/src/jolt/trace/rv.rs) for the mapping). Essentially, this is because different RISC-V instructions may have the same semantics (e.g. adding two operands together), with the only difference being what the operands represent (registers vs. immediates vs. program counter). For each Jolt instruction, we will specifically note if it is used to handle multiple RISC-V instructions.

## Notation

- `WORD_SIZE` (or $W$): The total bit size of the operands (either 32 or 64).

- $x$, $y$: The first and second input operands, respectively. For some instructions, there is only one operand, in which case the other operand is set to zero (and not involved in the lookup procedure). We read them in big-endian order.

- `C`: The number of chunks that each operand is split into. Current Jolt always sets $C = 4$ for $W = 32$ and $C = 8$ for $W = 64$.

- `M`: The size of the subtables (so that $m := \log_2(M) / 2$ is the number of bits in each operand of each chunk). Current Jolt always sets $M = 2 ^ {16}$ and thus $m = 8$.

## Helper Functions

We describe the helper functions used in implementing the instructions and in the specification below. These functions can be found in [instruction_utils.rs](../../../jolt-core/src/utils/instruction_utils.rs).

- **Truncation / Zero-extension:** Given word size $W \in \N$, we define the truncate function as:
    $$\begin{align*}
        \mathsf{Truncate}_{W}(z) = z \bmod 2^W \in \{0,1\}^{W}.
    \end{align*}$$
    Note that if $z$ has less than $W$ bits, it will be zero-extended to $W$ bits.

- **Sign-extension:** Given parameters $W,n \in \N$, we define the sign extension function as:
    $$\begin{align*}
        \mathsf{SignExtend}_{W,n}(z) =
            \begin{cases}
            (z \bmod 2^W) & \text{if } z_{n-1} = 0 \\
            (z \bmod 2^W) + \sum_{i=n}^{W-1} 2^i & \text{if } z_{n-1} = 1
            \end{cases}
    \end{align*}$$
    In other words, we take the $n$-th bit of $z$ to be the sign bit, and fill it in the remaining bits from $n$ to $W-1$. If $n > W$, this simply truncates $z$ to $W$ bits.

- **Chunking:** Given parameters $m, C \in \N$, we define the chunking function as:
    $$\begin{align*}
        \mathsf{Chunk}_{m,C}(z) = [z_0, z_1, \ldots, z_{C-1}] \in \left(\{0,1\}^{m}\right)^{C}.
    \end{align*}$$
    Here we write $\mathsf{Truncate}_{C \cdot m}(z) = z_0 \| z_1 \| \dots \| z_{C-1}$, with $z_i \in \{0,1\}^{m}$ for all $i$. Note that $z_0$ is the most significant chunk, and $z_{C-1}$ is the least significant chunk.

- **Chunk \& interleave:** Given parameters $m, C \in \N$, we define the chunk \& interleave function as:
    $$\begin{align*}
        \mathsf{ChunkInterleave}_{m,C}(x,y) = [x_0 \| y_0, \dots, x_{C-1} \| y_{C-1}] \in \left(\{0,1\}^{2m}\right)^{C},
    \end{align*}$$
    where $\mathsf{Chunk}_{m,C}(x) = [x_0, x_1, \ldots, x_{C-1}]$ and $\mathsf{Chunk}_{m,C}(y) = [y_0, y_1, \ldots, y_{C-1}]$.

- **Chunk for shift:** Given parameters $m, C \in \N$, we define the chunk for shift function as:
    $$\begin{align*}
        \mathsf{ChunkForShift}_{m,C}(x,y) = [x_0\| y_{C-1}, x_1\| y_{C-1}, \ldots, x_{C-1}\| y_{C-1}] \in \left(\{0,1\}^{2m}\right)^{C},
    \end{align*}$$
    where $\mathsf{Chunk}_{m,C}(x) = [x_0, x_1, \ldots, x_{C-1}]$ and $\mathsf{Chunk}_{m,C}(y) = [y_0, y_1, \ldots, y_{C-1}]$. This is similar to chunk \& interleave except the second half is always the last chunk of $y$.

- **Concatenate:** Given a sequence of words $(x_0, x_1, \ldots, x_{C-1}) \in \left(\{0,1\}^{m}\right)^{C}$, we define the concatenate function as:
    $$\begin{align*}
        \mathsf{Concatenate}_{m,C}(x_0, x_1, \ldots, x_{C-1}) &= \sum_{i=0}^{C-1} x_{C - 1 - i} \cdot 2^{i \cdot m} \\
        &= x_0 \| x_1 \| \dots \| x_{C-1} \in \{0,1\}^{C \cdot m}.
    \end{align*}$$

## Instructions List

### Logical & Arithmetic Instructions

<!-- markdownlint-disable MD029 -->

1. [`ANDInstruction`](../../../jolt-core/src/jolt/instruction/and.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected output:** Bitwise AND of two unsigned $W$-bit integers:
\[x \land y \in \{0,1\}^W.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{AND}}(x,y) = \mathsf{ChunkInterleave}_{\,m,C}(x,y).\]
- **Subtables:** $$\begin{align*}
    \mathsf{Subtables}_{\,\mathsf{AND}} = \left(\left[(\mathsf{And},i)\right]_{i=0}^{C-1}\right).
\end{align*}$$
- **Lookup combination:** Let $(Z_0, \ldots, Z_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{AND}}(Z_0, \ldots, Z_{C-1}) = \mathsf{Concatenate}_{\,m,C}(Z_0, \ldots, Z_{C-1}).\]

This is used for the following RISC-V instructions: `AND`, `ANDI`.

2. [`ORInstruction`](../../../jolt-core/src/jolt/instruction/or.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected output:** Bitwise OR of two unsigned $W$-bit integers:
\[x \lor y \in \{0,1\}^W.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{OR}}(x,y) = \mathsf{ChunkInterleave}_{\,m,C}(x,y).\]
- **Subtables:** $$\begin{align*}
    \mathsf{Subtables}_{\,\mathsf{OR}} = \left(\left[(\mathsf{Or},i)\right]_{i=0}^{C-1}\right).
\end{align*}$$
- **Lookup combination:** Let $(Z_0, \ldots, Z_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{OR}}(Z_0, \ldots, Z_{C-1}) = \mathsf{Concatenate}_{\,m,C}(Z_0, \ldots, Z_{C-1}).\]

This is used for the following RISC-V instructions: `OR`, `ORI`.

3. [`XORInstruction`](../../../jolt-core/src/jolt/instruction/xor.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected output:** Bitwise XOR of two unsigned $W$-bit integers:
\[x \oplus y \in \{0,1\}^W.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{XOR}}(x,y) = \mathsf{ChunkInterleave}_{\,m,C}(x,y).\]
- **Subtables:** $$\begin{align*}
    \mathsf{Subtables}_{\,\mathsf{XOR}} = \left(\left[(\mathsf{Xor},i)\right]_{i=0}^{C-1}\right).
\end{align*}$$
- **Lookup combination:** Let $(Z_0, \ldots, Z_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{XOR}}(Z_0, \ldots, Z_{C-1}) = \mathsf{Concatenate}_{\,m,C}(Z_0, \ldots, Z_{C-1}).\]

This is used for the following RISC-V instructions: `XOR`, `XORI`.

4. [`ADDInstruction`](../../../jolt-core/src/jolt/instruction/add.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected Output:** Addition of two unsigned $W$-bit integers, truncated to $W$ bits:
\[\mathsf{Truncate}_{\, W}(x + y) \in \{0,1\}^{W}.\]
- **Chunking:** Let $z = x + y \in \{0,1\}^{W+1}$ be the addition result without truncation. Then
\[\mathsf{Chunk}_{\,\mathsf{ADD}}(x,y) = \mathsf{Chunk}_{\, 2m,C}(z).\]
- **Subtables:** Let $k = C - W/(2m)$, which is $2$ for $W = 32$ and $4$ for $W = 64$. Then $$\begin{align*}
    \mathsf{Subtables}_{\,\mathsf{ADD}} = \left(\left[(\mathsf{TruncateOverflow}_{W},i)\right]_{i=0}^{k-1},\left[(\mathsf{Identity},i)\right]_{i=k}^{C - 1}\right).
\end{align*}$$
- **Lookup combination:** Let $(Z_0, \ldots, Z_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{ADD}}(Z_0, \ldots, Z_{C-1}) = \mathsf{Concatenate}_{\,m,C}(Z_0, \ldots, Z_{C-1}).\]

This is used for the following RISC-V instructions: `ADD`, `ADDI`, `JAL`, `JALR`, and `AUIPC`.

5. [`SUBInstruction`](../../../jolt-core/src/jolt/instruction/sub.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected Output:** Subtraction of two unsigned $W$-bit integers, with the result taken modulo $2^W$:
\[\mathsf{Truncate}_{\, W}(x - y) \in \{0,1\}^{W}.\]
- **Chunking:** Let $z = x + (2^W - y) \in \{0,1\}^{W+1}$ be the two's complement subtraction result. Then
\[\mathsf{Chunk}_{\,\mathsf{SUB}}(x,y) = \mathsf{Chunk}_{2m,C}(z).\]
- **Subtables:** Let $k = C - W/(2m)$, which is $2$ for $W = 32$ and $4$ for $W = 64$. Then $$\begin{align*}
    \mathsf{Subtables}_{\,\mathsf{SUB}} = \left(\left[(\mathsf{TruncateOverflow}_{W},i)\right]_{i=0}^{k-1},\left[(\mathsf{Identity},i)\right]_{i=k}^{C - 1}\right).
\end{align*}$$
- **Lookup combination:** Let $(Z_0, \ldots, Z_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{SUB}}(Z_0, \ldots, Z_{C-1}) = \mathsf{Concatenate}_{\,m,C}(Z_0, \ldots, Z_{C-1}).\]

6. [`MULInstruction`](../../../jolt-core/src/jolt/instruction/mul.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected Output:** Multiplication of two **signed** $W$-bit integers, truncated to $W$ bits:
\[\mathsf{Truncate}_{W}(x \cdot y) \in \{0,1\}^{W}.\]
- **Chunking:** Let $z = x \cdot y \in \{0,1\}^{2W}$ be the multiplication result without truncation. Then
\[\mathsf{Chunk}_{\,\mathsf{MUL}}(x,y) = \mathsf{Chunk}_{2m,C}(z).\]
- **Subtables:** Let $k = C - W/(2m)$. Then $$\begin{align*}
    \mathsf{Subtables}_{\,\mathsf{MUL}} = \left(\left[(\mathsf{TruncateOverflow}_{W},i)\right]_{i=0}^{k-1},\left[(\mathsf{Identity},i)\right]_{i=k}^{C - 1}\right).
\end{align*}$$
- **Lookup Combination:** Let $(Z_0, \ldots, Z_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{MUL}}(Z_0, \ldots, Z_{C-1}) = \mathsf{Concatenate}_{\,2m,C}(Z_0, \ldots, Z_{C-1}).\]

7. [`MULUInstruction`](../../../jolt-core/src/jolt/instruction/mul.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected Output:** Multiplication of two **unsigned** $W$-bit integers, truncated to $W$ bits:
\[\mathsf{Truncate}_{W}(x \cdot y) \in \{0,1\}^{W}.\]
- **Chunking:** Let $z = x \cdot y \in \{0,1\}^{2W}$ be the multiplication result without truncation. Then
\[\mathsf{Chunk}_{\,\mathsf{MULU}}(x,y) = \mathsf{Chunk}_{2m,C}(z).\]
- **Subtables:** Let $k = C - W/(2m)$. Then $$\begin{align*}
    \mathsf{Subtables}_{\,\mathsf{MULU}} = \left(\left[(\mathsf{TruncateOverflow}_{W},i)\right]_{i=0}^{k-1},\left[(\mathsf{Identity},i)\right]_{i=k}^{C - 1}\right).
\end{align*}$$
- **Lookup Combination:** Let $(Z_0, \ldots, Z_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{MULU}}(Z_0, \ldots, Z_{C-1}) = \mathsf{Concatenate}_{\,2m,C}(Z_0, \ldots, Z_{C-1}).\]

8. [`MULHUInstruction`](../../../jolt-core/src/jolt/instruction/mul.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected Output:** $\mathsf{Truncate}_{W}((x \cdot y) \gg W) \in \{0,1\}^{W}$, as high $W$-bit half of the multiplication of two **unsigned** $W$-bit integers.
- **Chunking:** Let $z = x \cdot y \in \{0,1\}^{2W}$ be the multiplication result. Then
\[\mathsf{Chunk}_{\,\mathsf{MULHU}}(x,y) = \mathsf{Chunk}_{2m,C}(z).\]
- **Subtables:** $$\begin{align*}
    \mathsf{Subtables}_{\,\mathsf{MULHU}} = \left(\left[(\mathsf{Identity},i)\right]_{i=0}^{C/2 - 1}\right).
\end{align*}$$
- **Lookup Combination:** Let $(Z_0, \ldots, Z_{C/2-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{MULHU}}(Z_0, \ldots, Z_{C/2-1}) = \mathsf{Concatenate}_{\,2m,C/2}(Z_0, \ldots, Z_{C/2-1}).\]

9. [`SLLInstruction`](../../../jolt-core/src/jolt/instruction/sll.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected Output:** The left shift of an unsigned $W$-bit integer $x$ by $(y \bmod W)$ bits, truncated to $W$ bits:
\[\mathsf{Truncate}_W(x \ll (y \bmod W)) \in \{0,1\}^{W}.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{SLL}}(x,y) = \mathsf{ChunkForShift}_{m,C}(x,y).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{SLL}} = \left(\left[(\mathsf{Sll}_{C-i,W},i)\right]_{i=0}^{C-1}\right).\]
- **Lookup Combination:** Let $(Z_0, \ldots, Z_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{SLL}}(Z_0, \ldots, Z_{C-1}) = \mathsf{Concatenate}_{\,2m,C}(Z_0, \ldots, Z_{C-1}).\]

This is used for the following RISC-V instructions: `SLL`, `SLLI`.

10. [`SRLInstruction`](../../../jolt-core/src/jolt/instruction/srl.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected Output:** The right shift of an unsigned $W$-bit integer $x$ by $(y \bmod W)$ bits, zero-extended to $W$ bits:
\[\mathsf{Truncate}_W(x \gg (y \bmod W)) \in \{0,1\}^{W}.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{SRL}}(x,y) = \mathsf{ChunkForShift}_{m,C}(x,y).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{SRL}} = \left(\left[(\mathsf{Srl}_{C-i,W},i)\right]_{i=0}^{C-1}\right).\]
- **Lookup Combination:** Let $(Z_0, \ldots, Z_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{SRL}}(Z_0, \ldots, Z_{C-1}) = Z_0 + \dots + Z_{C-1}.\]

This is used for the following RISC-V instructions: `SRL`, `SRLI`.

11. [`SRAInstruction`](../../../jolt-core/src/jolt/instruction/sra.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected Output:** The right shift of a signed $W$-bit integer $x$ by $(y \bmod W)$ bits, sign-extended to $W$ bits:
\[\mathsf{SignExtend}_W(\mathsf{SignExtend}_W(x) \gg (y \bmod W)) \in \{0,1\}^{W}.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{SRA}}(x,y) = \mathsf{ChunkForShift}_{m,C}(x,y).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{SRA}} = \left(\left[(\mathsf{Srl}_{C-i,W},i)\right]_{i=0}^{C-1}, \left(\mathsf{SraSign}_{W},0\right)\right).\]
- **Lookup Combination:** Let $(Z_0, \ldots, Z_{C-1}, S)$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{SRA}}(Z_0, \ldots, Z_{C-1}, S) = Z_0 + \dots + Z_{C-1} + S.\]

This is used for the following RISC-V instructions: `SRA`, `SRAI`.

### Comparison Instructions

12. [`BEQInstruction`](../../../jolt-core/src/jolt/instruction/beq.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected Output:** Whether two operands are equal: $(x \stackrel{?}{=} y) \in \{0,1\}$.
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{BEQ}}(x,y) = \mathsf{ChunkInterleave}_{m,C}(x,y).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{BEQ}} = \left(\left[(\mathsf{Eq},i)\right]_{i=0}^{C-1}\right).\]
- **Lookup Combination:** Let $(Z_0, \ldots, Z_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{BEQ}}(Z_0, \ldots, Z_{C-1}) = \prod_{i=0}^{C-1} Z_i.\]

13. [`BNEInstruction`](../../../jolt-core/src/jolt/instruction/bne.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected Output:** Whether two operands are not equal:
\[(x \stackrel{?}{\ne} y) \in \{0,1\}.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{BNE}}(x,y) = \mathsf{ChunkInterleave}_{m,C}(x,y).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{BNE}} = \left(\left[(\mathsf{Eq},i)\right]_{i=0}^{C-1}\right).\]
- **Lookup Combination:** Let $(Z_0, \ldots, Z_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{BNE}}(Z_0, \ldots, Z_{C-1}) = 1 - \prod_{i=0}^{C-1} Z_i.\]

14. [`SLTUInstruction`](../../../jolt-core/src/jolt/instruction/sltu.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected Output:** Whether the first operand is less than the second operand (both interpreted as unsigned $W$-bit numbers):
\[(x \stackrel{?}{<} y) \in \{0,1\}.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{SLTU}}(x,y) = \mathsf{ChunkInterleave}_{m,C}(x,y).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{SLTU}} = \left(\left[(\mathsf{Ltu},i)\right]_{i=0}^{C-1},\left[(\mathsf{Eq},i)\right]_{i=0}^{C-2}\right).\]
- **Lookup Combination:** Let $(Z_0, \ldots, Z_{C-1},W_0, \ldots, W_{C-2})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{SLTU}}(Z_0, \ldots, Z_{C-1},W_0, \ldots, W_{C-2}) = \sum_{i=0}^{C-1} Z_i \cdot \prod_{j=0}^{i-1} W_j.\]

This is used for the following RISC-V instructions: `SLTU`, `SLTIU`.

15. [`BGEUInstruction`](../../../jolt-core/src/jolt/instruction/bgeu.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected Output:** Whether the first operand is greater than or equal to the second operand (both interpreted as unsigned $W$-bit numbers):
\[(x \stackrel{?}{\ge} y) \in \{0,1\}.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{BGEU}}(x,y) = \mathsf{ChunkInterleave}_{m,C}(x,y).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{BGEU}} = \left(\left[(\mathsf{Ltu},i)\right]_{i=0}^{C-1},\left[(\mathsf{Eq},i)\right]_{i=0}^{C-2}\right).\]
- **Lookup Combination:** Let $(Z_0, \ldots, Z_{C-1},W_0, \ldots, W_{C-2})$ be the lookup results. Then
\[\begin{align*}
    &\quad \mathsf{Combine}_{\,\mathsf{BGEU}}(Z_0, \ldots, Z_{C-1},W_0, \ldots, W_{C-2}) \\
    &= 1 - \mathsf{Combine}_{\,\mathsf{SLTU}}(Z_0, \ldots, Z_{C-1},W_0, \ldots, W_{C-2}).
\end{align*}\]

16. [`SLTInstruction`](../../../jolt-core/src/jolt/instruction/slt.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected Output:** Whether the first operand is less than the second operand (both interpreted as **signed** $W$-bit numbers):
\[(x \stackrel{?}{<} y) \in \{0,1\}.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{SLT}}(x,y) = \mathsf{ChunkInterleave}_{m,C}(x,y).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{SLT}} = \left(\begin{gathered} (\mathsf{LeftMSB},0), (\mathsf{RightMSB},0),\\
            \left[(\mathsf{Ltu},i)\right]_{i=1}^{C-1},\left[(\mathsf{Eq},i)\right]_{i=1}^{C-2},\\
            (\mathsf{LtAbs},0), (\mathsf{EqAbs},0)\end{gathered}\right).\]
- **Lookup Combination:** Let $(L,R,Z_1,\dots,Z_{C-1},W_1,\dots,W_{C-2},Z_0,W_0)$ be the lookup results. Then
\[\begin{align*}
    &\quad \mathsf{Combine}_{\,\mathsf{SLT}}(L,R,Z_1,\dots,Z_{C-1},W_1,\dots,W_{C-2},Z_0,W_0) \\
    &= L \cdot (1 - R) + \left((1 - L) \cdot (1 - R) + L \cdot R\right) \cdot \left(\sum_{i=0}^{C-1} Z_i \cdot \prod_{j=0}^{i-1} W_j\right).
\end{align*}\]

This is used for the following RISC-V instructions: `SLT`, `SLTI`.

17. [`BGEInstruction`](../../../jolt-core/src/jolt/instruction/bge.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected Output:** Whether the first operand is greater than or equal to the second operand (both interpreted as **signed** $W$-bit numbers):
\[(x \stackrel{?}{\ge} y) \in \{0,1\}.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{BGE}}(x,y) = \mathsf{ChunkInterleave}_{m,C}(x,y).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{BGE}} = \left(\begin{gathered} (\mathsf{LeftMSB},0), (\mathsf{RightMSB},0),\\
                \left[(\mathsf{Ltu},i)\right]_{i=1}^{C-1},\left[(\mathsf{Eq},i)\right]_{i=1}^{C-2},\\
                (\mathsf{LtAbs},0), (\mathsf{EqAbs},0)\end{gathered}\right).\]
- **Lookup Combination:** Let $(L,R,Z_1,\dots,Z_{C-1},W_1,\dots,W_{C-2},Z_0,W_0)$ be the lookup results. Then
\[\begin{align*}
    &\quad \mathsf{Combine}_{\,\mathsf{BGE}}(L,R,Z_1,\dots,Z_{C-1},W_1,\dots,W_{C-2},Z_0,W_0) \\
    &= 1 - \mathsf{Combine}_{\,\mathsf{SLT}}(L,R,Z_1,\dots,Z_{C-1},W_1,\dots,W_{C-2},Z_0,W_0).
\end{align*}\]

### Load & Store Instructions

18. [`LBInstruction`](../../../jolt-core/src/jolt/instruction/lb.rs)

- **Operands:** $x \in \{0,1\}^W$
- **Expected Output:** the lower 8 bits of $x$ sign-extended to $W$ bits:
\[\mathsf{SignExtend}_W(x \bmod 2^8) \in \{0,1\}^W.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{LB}}(x) = \mathsf{Chunk}_{m,C}(x).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{LB}} = \left(\begin{gathered}
    (\mathsf{TruncateOverflow}_{8},C-1), (\mathsf{SignExtend}_{8},C-1), \\
     \left[(\mathsf{Identity},i)\right]_{i=0}^{C-1}
\end{gathered}\right).\]
- **Lookup Combination:** Let $(Z, S, I_0, \ldots, I_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{LB}}(Z, S, I_0, \ldots, I_{C-1}) = Z + \sum_{i=1}^{C-1} 2^{8 \cdot i} \cdot S.\]

19. [`LHInstruction`](../../../jolt-core/src/jolt/instruction/lh.rs)

- **Operands:** $x \in \{0,1\}^W$
- **Expected Output:** the lower 16 bits of $x$ sign-extended to $W$ bits:
\[\mathsf{SignExtend}_W(x \bmod 2^{16}) \in \{0,1\}^W.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{LH}}(x) = \mathsf{Chunk}_{m,C}(x).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{LH}} = \left((\mathsf{Identity},C-1), (\mathsf{SignExtend},C-1), \left[(\mathsf{Identity},i)\right]_{i=0}^{C-1}\right).\]
- **Lookup Combination:** Let $(Z, S, I_0, \ldots, I_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{LH}}(Z, S, I_0, \ldots, I_{C-1}) = Z + \sum_{i=0}^{C-1} 2^{16 \cdot i} \cdot S.\]

20. [`SBInstruction`](../../../jolt-core/src/jolt/instruction/sb.rs)

- **Operands:** $x \in \{0,1\}^W$
- **Expected Output:** The lower 8 bits of $x$ zero-extended to $W$ bits:
\[\mathsf{ZeroExtend}_W(x \bmod 2^8) \in \{0,1\}^W.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{SB}}(x) = \mathsf{Chunk}_{m,C}(x).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{SB}} = \left((\mathsf{TruncateOverflow}_{8},C-1), \left[(\mathsf{Identity},i)\right]_{i=0}^{C-1}\right).\]
- **Lookup Combination:** Let $(Z, I_0, \ldots, I_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{SB}}(Z, I_0, \ldots, I_{C-1}) = Z.\]

21. [`SHInstruction`](../../../jolt-core/src/jolt/instruction/sh.rs)

- **Operands:** $x \in \{0,1\}^W$
- **Expected Output:** The lower 16 bits of $x$ zero-extended to $W$ bits:
\[\mathsf{ZeroExtend}_W(x \bmod 2^{16}) \in \{0,1\}^W.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{SH}}(x) = \mathsf{Chunk}_{m,C}(x).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{SH}} = \left((\mathsf{Identity},C-1), \left[(\mathsf{Identity},i)\right]_{i=0}^{C-1}\right).\]
- **Lookup Combination:** Let $(I_0, \ldots, I_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{SH}}(I_0, \ldots, I_{C-1}) = I_0.\]

22. [`SWInstruction`](../../../jolt-core/src/jolt/instruction/sw.rs)

- **Operands:** $x \in \{0,1\}^W$
- **Expected Output:** The lower 32 bits of $x$ zero-extended to $W$ bits:
\[\mathsf{ZeroExtend}_W(x \bmod 2^{32}) \in \{0,1\}^W.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{SW}}(x) = \mathsf{Chunk}_{m,C}(x).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{SW}} = \left((\mathsf{Identity},C-2), (\mathsf{Identity},C-1)\right).\]
- **Lookup Combination:** Let $(I_{C-2}, I_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{SW}}(I_{C-2}, I_{C-1}) = \mathsf{Concatenate}_{m, 2}(I_{C-2}, I_{C-1}).\]

### Virtual Instructions

23. [`ADVICEInstruction`](../../../jolt-core/src/jolt/instruction/virtual_advice.rs)

- **Operands:** $x \in \{0,1\}^{W}$
- **Expected Output:** $x$.
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{ADVICE}}(x) = \mathsf{Chunk}_{2m,C}(x).\]
- **Subtables:** Let $k = C - W/(2m)$. Return
\[\begin{align*}
    \mathsf{Subtables}_{\,\mathsf{ADVICE}} = \left(\left[(\mathsf{TruncateOverflow}_{W},i)\right]_{i=0}^{k-1},\left[(\mathsf{Identity},i)\right]_{i=k}^{C - 1}\right).
\end{align*}\]
- **Lookup Combination:** Let $(Z_0, \ldots, Z_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{ADVICE}}(Z_0, \ldots, Z_{C-1}) = \mathsf{Concatenate}_{2m,C}(Z_0, \ldots, Z_{C-1}).\]

24. [`ASSERTLTEInstruction`](../../../jolt-core/src/jolt/instruction/virtual_assert_lte.rs)

- **Operands:** $x, y \in \{0,1\}^{W}$
- **Expected Output:** Whether the first operand is less than or equal to the second operand (both interpreted as **unsigned** $W$-bit numbers):
\[(x \stackrel{?}{\le} y) \in \{0,1\}.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{ASSERTLTE}}(x,y) = \mathsf{ChunkInterleave}_{m,C}(x,y).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{ASSERTLTE}} = \left(\left[(\mathsf{Ltu},i)\right]_{i=0}^{C-1},\left[(\mathsf{Eq},i)\right]_{i=0}^{C-1}\right).\]
- **Lookup Combination:** Let $(Z_0, \ldots, Z_{C-1},W_0, \ldots, W_{C-1})$ be the lookup results. Then
\[\begin{align*}
    &\quad \mathsf{Combine}_{\,\mathsf{ASSERTLTE}}(Z_0, \ldots, Z_{C-1},W_0, \ldots, W_{C-1}) \\
    &= \mathsf{Combine}_{\,\mathsf{SLTU}}(Z_0, \ldots, Z_{C-1},W_0, \ldots, W_{C-2}) + \mathsf{Combine}_{\,\mathsf{BEQ}}(W_{0}, \ldots, W_{C-1})\\
    &= \sum_{i=0}^{C-1} Z_i \cdot \prod_{j=0}^{i-1} W_j + \prod_{i=0}^{C-1} W_i.
\end{align*}\]

25. [`AssertValidDiv0Instruction`](../../../jolt-core/src/jolt/instruction/virtual_assert_valid_div0.rs)

- **Operands:** $x, y \in \{0,1\}^{W}$
- **Expected Output:** Return $1$ if either the first operand (considered as the divisor) is non-zero, or the first operand is zero and the second operand (considered as the quotient) is $11\ldots 1$ (interpreted as an unsigned $W$-bit number); otherwise return $0$:
\[(\lnot (x \stackrel{?}{=} 0) \lor (y \stackrel{?}{=} 2^W - 1)) \in \{0,1\}.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{AssertValidDiv0}}(x,y) = \mathsf{ChunkInterleave}_{m,C}(x,y).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{AssertValidDiv0}} = \left(\left[(\mathsf{LeftIsZero},i)\right]_{i=0}^{C-1},\left[(\mathsf{DivByZero},i)\right]_{i=0}^{C-1}\right).\]
- **Lookup Combination:** Let $(Z_0, \ldots, Z_{C-1},W_0, \ldots, W_{C-1})$ be the lookup results. Then
\[\begin{align*}
    \mathsf{Combine}_{\,\mathsf{AssertValidDiv0}}(Z_0, \ldots, Z_{C-1},W_0, \ldots, W_{C-1})
    &= 1 - \prod_{i=0}^{C-1} Z_i + \prod_{i=0}^{C-1} W_i.
\end{align*}\]

26. [`AssertValidSignedRemainderInstruction`](../../../jolt-core/src/jolt/instruction/virtual_assert_valid_signed_remainder.rs)

- **Operands:** $x, y \in \{0,1\}^{W}$
- **Expected Output:** Interpret the operands as **signed** $W$-bit numbers, with the first operand being the remainder and the second being the divisor. Return $1$ if either the remainder or the divisor is zero, or the sign of the remainder is less than the sign of the divisor and also that the non-sign part of the remainder and the divisor are equal; otherwise return $0$:
\[(x \stackrel{?}{=} 0) \lor (y \stackrel{?}{=} 0) \lor ((x_0 \stackrel{?}{<} y_0) \land (x_{[1:W-1]} \stackrel{?}{=} y_{[1:W-1]})) \in \{0,1\}.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{AssertValidSignedRemainder}}(x,y) = \mathsf{ChunkInterleave}_{m,C}(x,y).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{AssertValidSignedRemainder}} = \left(\begin{gathered}
    (\mathsf{LeftMSB},0), (\mathsf{RightMSB},0),\\
    \left[(\mathsf{Eq},i)\right]_{i=1}^{C-1}, \left[(\mathsf{Ltu},i)\right]_{i=1}^{C-1},\\
    (\mathsf{EqAbs},0), (\mathsf{LtAbs},0),\\
    \left[(\mathsf{LeftIsZero},i)\right]_{i=0}^{C-1},\\
    \left[(\mathsf{RightIsZero},i)\right]_{i=0}^{C-1}
\end{gathered}\right).\]
- **Lookup Combination:** Let the lookup results be \[(L, R, Z_1, \ldots, Z_{C-1},W_1, \ldots, W_{C-2},Z_0,W_0,L'_0,\dots,L'_{C-1},R'_0,\dots,R'_{C-1}).\] Then
\[\begin{align*}
    &\quad \mathsf{Combine}_{\,\mathsf{AssertValidSignedRemainder}}(Z_0, \ldots, Z_{C-1},W_0, \ldots, W_{C-1}) \\
    &= (1 - L - R) \cdot \mathsf{Combine}_{\,\mathsf{SLTU}}(Z_0, \ldots, Z_{C-1},W_0, \ldots, W_{C-2}) \\
    &\qquad + L \cdot R \cdot (1 - \mathsf{Combine}_{\,\mathsf{BEQ}}(W_0, \ldots, W_{C-1}))\\
    &\qquad + (1 - L) \cdot R \cdot \prod_{i=0}^{C-1} L'_i + \prod_{i=0}^{C-1} R'_i
\end{align*}\]

27. [`AssertValidUnsignedRemainderInstruction`](../../../jolt-core/src/jolt/instruction/virtual_assert_valid_unsigned_remainder.rs)

- **Operands:** $x, y \in \{0,1\}^{W}$
- **Expected Output:** Interpret the operands as **unsigned** $W$-bit numbers, with the first operand being the remainder and the second being the divisor. Return $1$ if either the divisor is zero, or the remainder is less than the divisor; otherwise return $0$:
\[(y \stackrel{?}{=} 0) \lor (x \stackrel{?}{<} y) \in \{0,1\}.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{AssertValidUnsignedRemainder}}(x,y) = \mathsf{ChunkInterleave}_{m,C}(x,y).\]
- **Subtables:**
\[\mathsf{Subtables}_{\,\mathsf{AssertValidUnsignedRemainder}} = \left(\begin{gathered}
    \left[(\mathsf{Ltu},i)\right]_{i=0}^{C-1}, \left[(\mathsf{Eq},i)\right]_{i=0}^{C-2},\\
    \left[(\mathsf{RightIsZero},i)\right]_{i=0}^{C-1}
\end{gathered}\right).\]
- **Lookup Combination:** Let the lookup results be \[(Z_0, \ldots, Z_{C-1},W_0, \ldots, W_{C-2}, R_0, \ldots, R_{C-1}).\] Then
\[\begin{align*}
    &\quad \mathsf{Combine}_{\,\mathsf{AssertValidUnsignedRemainder}}(Z_0, \ldots, Z_{C-1},W_0, \ldots, W_{C-2}, R_0, \ldots, R_{C-1}) \\
    &= \mathsf{Combine}_{\,\mathsf{SLTU}}(Z_0, \ldots, Z_{C-1},W_0, \ldots, W_{C-2}) + \prod_{i=0}^{C-1} R_i.
\end{align*}\]

28. [`MOVEInstruction`](../../../jolt-core/src/jolt/instruction/virtual_move.rs)

- **Operands:** $x \in \{0,1\}^{W}$
- **Expected Output:** $x$.
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{MOVE}}(x) = \mathsf{Chunk}_{2m,C}(x).\]
- **Subtables:** Return
\[\mathsf{Subtables}_{\,\mathsf{MOVE}} = \left(\left[(\mathsf{Identity},i)\right]_{i=0}^{C-1}\right).\]
- **Lookup Combination:** Let $(Z_0, \ldots, Z_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{MOVE}}(Z_0, \ldots, Z_{C-1}) = \mathsf{Concatenate}_{2m,C}(Z_0, \ldots, Z_{C-1}).\]

<!-- This virtual instruction is used in `DIV`, `DIVU`, `REM`, and `REMU`. -->

29. [`MOVSIGNInstruction`](../../../jolt-core/src/jolt/instruction/virtual_movsign.rs)

- **Operands:** $x \in \{0,1\}^{W}$
- **Expected Output:** Interpret $x$ as a **signed** $W$-bit number, and return an unsigned $W$-bit number with the sign bit of $x$ extended to all bits of the result:
\[x_0 \cdot (2^W - 1),\qquad \text{ where } x_0\text{ is the sign bit of } x.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{MOVSIGN}}(x) = \mathsf{Chunk}_{2m,C}(x).\]
- **Subtables:** Let $k = C - W/(2m)$. Return
\[\mathsf{Subtables}_{\,\mathsf{MOVSIGN}} = \left(\left[(\mathsf{SignExtend}_{16},k)\right], \left[(\mathsf{Identity},i)\right]_{i=0}^{C-1}\right).\]
- **Lookup Combination:** Let $(S, Z_0,\ldots, Z_{W/(2m) - 1})$ be the lookup results. Then
\[\begin{align*}
    \mathsf{Combine}_{\,\mathsf{MOVSIGN}}(S, Z_0, \ldots, Z_{W/(2m) - 1}) = \mathsf{Concatenate}_{2m, 1 + W/(2m)}(S, S, \dots, S).
\end{align*}\]

<!-- This virtual instruction is used in `MULH` and `MULHSU`. -->

### Multi-Cycle Instructions

See [M extension](../spec/m_extension.md) for the description of the multi-cycle instructions (MULH, MULHSU, DIV, DIVU, REM, REMU). Each of these instructions is implemented using a sequence of (single-cycle) instructions described above.

<!--
#### `MULH`

Expected output: the high bits of $x * y$ where $x$ is signed and $y$ is signed.

Let `r_x := rs1`, `r_y := rs2`, `x := rs1_val`, `y := rs2_val`. Initialize virtual registers `v_sx := 32`, `v_sy := 33`, `v_0 := 34`, `v_1 := 35`, `v_2 := 36`, `v_3 := 37`.

1. Let `s_x` be the value of applying `MOVSIGNInstruction` to `x`. 

First virtual instruction is `VIRTUAL_MOVSIGN` with `rs1 := r_x`, `rd := v_sx`, `rs1_val := x`, `rd_post_val := s_x`.

2. Let `s_y` be the value of applying `MOVSIGNInstruction` to `y`.

Second virtual instruction is `VIRTUAL_MOVSIGN` with `rs1 := r_y`, `rd := v_sy`, `rs1_val := y`, `rd_post_val := s_y`.

3. Let `xy_high_bits` be the value of applying `MULHUInstruction` to `x` and `y`.

Third virtual instruction is `MULHU` with `rs1 := r_x`, `rs2 := r_y`, `rd := v_0`, `rs1_val := x`, `rs2_val := y`, `rd_post_val := xy_high_bits`.

4. Let `sx_y_low_bits` be the value of applying `MULUInstruction` to `s_x` and `y`.

Fourth virtual instruction is `MULU` with `rs1 := v_sx`, `rs2 := r_y`, `rd := v_1`, `rs1_val := s_x`, `rs2_val := y`, `rd_post_val := sx_y_low_bits`.

5. Let `sy_x_low_bits` be the value of applying `MULUInstruction` to `s_y` and `s_x`.

Fifth virtual instruction is `MULU` with `rs1 := v_sy`, `rs2 := r_x`, `rd := v_2`, `rs1_val := s_y`, `rs2_val := s_x`, `rd_post_val := sy_x_low_bits`.

6. Let `partial_sum` be the value of applying `ADDInstruction` to `xy_high_bits` and `sx_y_low_bits`.

Sixth virtual instruction is `ADD` with `rs1 := v_0`, `rs2 := v_1`, `rd := v_3`, `rs1_val := xy_high_bits`, `rs2_val := sy_x_low_bits`, `rd_post_val := partial_sum`.

7. Let `result` be the value of applying `ADDInstruction` to `partial_sum` and `sy_x_low_bits`.

Seventh virtual instruction is `ADD` with `rs1 := v_3`, `rs2 := v_2`, `rd := rd` (original `rd`), `rs1_val := partial_sum`, `rs2_val := sy_x_low_bits`, `rd_post_val := result`.

#### `MULHSU`

Expected output: the high bits of $x * y$ where $x$ is signed and $y$ is unsigned. (double-check?)

Let `r_x := rs1`, `r_y := rs2`, `x := rs1_val`, `y := rs2_val`. Initialize virtual registers `v_sx := 32`, `v_1 := 33`, `v_2 := 34`, `v_3 := 35`.

1. Let `s_x` be the result of applying `MOVSIGNInstruction` to `x`.

First virtual instruction is `VIRTUAL_MOVSIGN` with `rs1 := r_x`, `rd := v_sx`, `rs1_val := x`, `rd_post_val := s_x`.

2. Let `xy_high_bits` be the result of applying `MULHUInstruction` to `x` and `y`.

Second virtual instruction is `MULHU` with `rs1 := r_x`, `rs2 := r_y`, `rd := v_1`, `rs1_val := x`, `rs2_val := y`, `rd_post_val := xy_high_bits`.

3. Let `sx_y_low_bits` be the result of applying `MULUInstruction` to `s_x` and `y`.

Third virtual instruction is `MULU` with `rs1 := v_sx`, `rs2 := r_y`, `rd := v_2`, `rs1_val := s_x`, `rs2_val := y`, `rd_post_val := sx_y_low_bits`.

4. Let `result` be the result of applying `ADDInstruction` to `xy_high_bits` and `sx_y_low_bits`.

Fourth virtual instruction is `ADD` with `rs1 := v_1`, `rs2 := v_2`, `rd := rd` (original `rd`), `rs1_val := xy_high_bits`, `rs2_val := sx_y_low_bits`, `rd_post_val := result`.


#### `DIV`

Expected output: the quotient of $x / y$ where $x$ is signed and $y$ is signed, rounded towards zero.

Let `r_x := rs1`, `r_y := rs2`, `x := rs1_val`, `y := rs2_val`. Initialize virtual registers `v_0 := 32`, `v_q := 33`, `v_r := 34`, `v_qy := 35`.

Let `quotient` and `remainder` be the result of applying division on `x` and `y`, where both are signed.

1. Let `q` be the value of applying `ADVICEInstruction` to `quotient` (i.e. just range-check `quotient` to be `u32/u64`).

First virtual instruction is `VIRTUAL_ADVICE` with `rd := v_q`, `rd_post_val := v_q`, and `advice_value := quotient`.

(double-check on advice value)

2. Let `r` be the value of applying `ADVICEInstruction` to `remainder` (i.e. just range-check `remainder` to be `u32/u64`).

Second virtual instruction is `VIRTUAL_ADVICE` with `rd := v_r`, `rd_post_val := v_r`, and `advice_value := remainder`.

(double-check on advice value)

3. Let `is_valid` be the value of applying `AssertValidSignedRemainderInstruction` to `r` and `y`. Fails if `is_valid` is 0.

Third virtual instruction is `VIRTUAL_ASSERT_VALID_SIGNED_REMAINDER` with `rs1 := v_r`, `rs2 := r_y`, `rs1_val := r`, `rs2_val := y`.

4. Let `is_valid` be the value of applying `AssertValidDiv0Instruction` to `y` and `q`. Fails if `is_valid` is 0.

(Note: this overrides the previous `is_valid` value)

Fourth virtual instruction is `VIRTUAL_ASSERT_VALID_DIV0` with `rs1 := r_y`, `rs2 := v_q`, `rs1_val := y`, `rs2_val := q`.

5. Let `q_y` be the value of applying `MULInstruction` to `q` and `y`.

Fifth virtual instruction is `MUL` with `rs1 := v_q`, `rs2 := r_y`, `rd := v_qy`, `rs1_val := q`, `rs2_val := y`, `rd_post_val := q_y`.

6. Let `add_0` be the value of applying `ADDInstruction` to `q_y` and `r`.

Sixth virtual instruction is `ADD` with `rs1 := v_qy`, `rs2 := v_r`, `rd := v_0`, `rs1_val := q_y`, `rs2_val := r`, `rd_post_val := add_0`.

7. Let `_assert_eq` be the value of applying `BEQInstruction` to `add_0` and `r`.

(todo: put assert command that `_assert_eq` is 1 in https://github.com/a16z/jolt/blob/main/jolt-core/src/jolt/instruction/div.rs#L184)

Seventh virtual instruction is `VIRTUAL_ASSERT_EQ` with `rs1 := v_0`, `rs2 := r_x`, `rs1_val := add_0`, `rs2_val := r`.

8. Eighth virtual instruction is `VIRTUAL_MOVE` with `rs1 := v_q`, `rd := rd` (original `rd`), `rs1_val := q`, `rd_post_val := q`.


#### `DIVU`

Expected output: the quotient $x / y$ where $x$ is unsigned and $y$ is unsigned. If $y = 0$, the quotient is $2^{\verb|WORD_SIZE|} - 1$.

Let `r_x := rs1`, `r_y := rs2`, `x := rs1_val`, `y := rs2_val`. Initialize virtual registers `v_0 := 32`, `v_q := 33`, `v_r := 34`, `v_qy := 35`.

Let `quotient` and `remainder` be the result of applying division on `x` and `y`, where both are unsigned. If $y = 0$, then `quotient` is $2^{\verb|WORD_SIZE|} - 1$ and `remainder` is $x$.

1. Let `q` be the value of applying `ADVICEInstruction` to `quotient` (i.e. just range-check `quotient` to be `u32/u64`).

First virtual instruction is `VIRTUAL_ADVICE` with `rd := v_q`, `rd_post_val := q`, and `advice_value := quotient`.

(double-check on advice value)

2. Let `r` be the value of applying `ADVICEInstruction` to `remainder` (i.e. just range-check `remainder` to be `u32/u64`).

Second virtual instruction is `VIRTUAL_ADVICE` with `rd := v_r`, `rd_post_val := r`, and `advice_value := remainder`.

(double-check on advice value)

3. Let `q_y` be the value of applying `MULUInstruction` to `q` and `y`.

Third virtual instruction is `MULU` with `rs1 := v_q`, `rs2 := r_y`, `rd := v_qy`, `rs1_val := q`, `rs2_val := y`, `rd_post_val := q_y`.

4. Let `is_valid` be the value of applying `AssertValidUnsignedRemainderInstruction` to `r` and `y`.

Fourth virtual instruction is `VIRTUAL_ASSERT_VALID_UNSIGNED_REMAINDER` with `rs1 := v_r`, `rs2 := r_y`, `rs1_val := r`, `rs2_val := y`.

5. 6. 7. 8. 9. TODO

#### `REM`

(7 instructions)

#### `REMU`

(8 instructions)

Note for Div and Rem instructions: we cannot remove the final `MOVE` instruction (instead putting the quotient or remainder directly in the `rd` register) because of the edge case that `rd` is equal to `rs1` or `rs2`.

-->
