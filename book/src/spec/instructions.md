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

Note that the RISC-V instructions are not the same as the Jolt instructions. In fact, several RISC-V instructions may invoke the same Jolt instruction, or a single RISC-V instruction may invoke multiple Jolt instructions.

## Notation

- `WORD_SIZE` (or `W`): The total bit size of the operands (either 32 or 64).

- `x`, `y`: The first and second input operands, respectively. For some instructions, there is only one operand, in which case the other operand is set to zero (and not involved in the lookup procedure).

- `C`: The number of chunks that each operand is split into. Current Jolt always sets `C = 4` for `WORD_SIZE = 32` and `C = 8` for `WORD_SIZE = 64`.

- `M`: The size of the subtables (so that `m := \log_2(M) / 2` is the number of bits in each operand of each chunk). Current Jolt always sets `M = 2 ^ 16` and thus `m = 8`.

## Helper Functions

- Truncation / Zero-extension: For a `WORD_SIZE`-bit number `x`, `truncate_to_W(x)` returns the `WORD_SIZE`-bit number `x` itself.

- Sign-extension: For a `WORD_SIZE`-bit number `x`, `sign_extend_to_W(x)` returns the `WORD_SIZE`-bit number `x` itself.

- Chunking:

- Chunk & interleave:

- Chunk for shift:

- Concatenate:

## Instructions List

### Logical & Arithmetic Instructions

1. [`ANDInstruction`](../../../jolt-core/src/jolt/instruction/and.rs)

- **Operands:** $x, y \in \{0,1\}^W$
- **Expected output:** Bitwise AND of two unsigned $W$-bit integers:
\[x \land y \in \{0,1\}^W.\]
- **Chunking:**
\[\mathsf{Chunk}_{\,\mathsf{AND}}(x,y) = \mathsf{ChunkInterleave}_{\,m,C}(x,y).\]
- **Subtables:**
\begin{align*}
    \mathsf{Subtables}_{\,\mathsf{AND}} = \left(\left[(\mathsf{And},i)\right]_{i=0}^{C-1}\right).
\end{align*}
- **Lookup combination:** Let $(Z_0, \ldots, Z_{C-1})$ be the lookup results. Then
\[\mathsf{Combine}_{\,\mathsf{AND}}(Z_0, \ldots, Z_{C-1}) = \mathsf{Concatenate}_{\,m,C}(Z_0, \ldots, Z_{C-1}).\]

2. [`ORInstruction`](../../../jolt-core/src/jolt/instruction/or.rs)

3. [`XORInstruction`](../../../jolt-core/src/jolt/instruction/xor.rs)

4. [`ADDInstruction`](../../../jolt-core/src/jolt/instruction/add.rs)

5. [`SUBInstruction`](../../../jolt-core/src/jolt/instruction/sub.rs)

6. [`MULInstruction`](../../../jolt-core/src/jolt/instruction/mul.rs)

7. [`MULUInstruction`](../../../jolt-core/src/jolt/instruction/mul.rs)

8. [`MULHUInstruction`](../../../jolt-core/src/jolt/instruction/mul.rs)

9. [`SLLInstruction`](../../../jolt-core/src/jolt/instruction/sll.rs)

10. [`SRLInstruction`](../../../jolt-core/src/jolt/instruction/srl.rs)

11. [`SRAInstruction`](../../../jolt-core/src/jolt/instruction/sra.rs)

### Comparison Instructions

12. [`BEQInstruction`](../../../jolt-core/src/jolt/instruction/beq.rs)

13. [`BNEInstruction`](../../../jolt-core/src/jolt/instruction/bne.rs)

14. [`SLTInstruction`](../../../jolt-core/src/jolt/instruction/slt.rs)

15. [`SLTUInstruction`](../../../jolt-core/src/jolt/instruction/sltu.rs)

16. [`BGEInstruction`](../../../jolt-core/src/jolt/instruction/bge.rs)

17. [`BGEUInstruction`](../../../jolt-core/src/jolt/instruction/bgeu.rs)

### Load & Store Instructions

18. [`LBInstruction`](../../../jolt-core/src/jolt/instruction/lb.rs)

19. [`LHInstruction`](../../../jolt-core/src/jolt/instruction/lh.rs)

20. [`SBInstruction`](../../../jolt-core/src/jolt/instruction/sb.rs)

21. [`SHInstruction`](../../../jolt-core/src/jolt/instruction/sh.rs)

22. [`SWInstruction`](../../../jolt-core/src/jolt/instruction/sw.rs)

- Is constrained via two `IdentitySubtable`s on the third and fourth chunks of the second operand (for 32 bits only, what about 64?)

- What about the other chunks? Why can they be unconstrained? The answer is that the other chunks are simply ignored. There's no subtable lookup for them, so they are not present when combining lookups.

### Virtual Instructions

23. [`VirtualAdvice`](../../../jolt-core/src/jolt/instruction/virtual_advice.rs)

- Only range-check the advice to be `u32/u64`

24. [`VirtualAssertLTE`](../../../jolt-core/src/jolt/instruction/virtual_assert_lte.rs)

- Compute $x \le y$ as a combination of strict less-than and equality

25. [`VirtualAssertValidDiv0`](../../../jolt-core/src/jolt/instruction/virtual_assert_valid_div0.rs)

- Inputs $x, y$ are interpreted as (unsigned) divisor and quotient
- Output 1 if $x \ne 0$ or if $x = 0$ and $y = 2 ^ \verb|WORD_SIZE| - 1$

26. [`VirtualAssertValidSignedRemainder`](../../../jolt-core/src/jolt/instruction/virtual_assert_valid_signed_remainder.rs)

- First operand is the remainder, second is the divisor

- Output 1 if the remainder is valid w.r.t. the divisor, and 0 otherwise. Validity here means that either the remainder or divisor is 0, or they have the same sign and the remainder is less than the divisor in absolute value.

- This is checked by the subtables via getting the sign bits, the (equal 0) bit, and the (less than) comparison via subtables.

27. [`VirtualAssertValidUnsignedRemainder`](../../../jolt-core/src/jolt/instruction/virtual_assert_valid_unsigned_remainder.rs)

- First operand is the remainder, second is the divisor (both are unsigned)

- Output 1 if the remainder is valid w.r.t. the divisor, and 0 otherwise. Validity here means that either the divisor is 0, or the remainder is less than the divisor.

- This is checked by the subtables via getting the (equal 0) bit, and the (less than) comparison via subtables.

28. [`VirtualMove`](../../../jolt-core/src/jolt/instruction/virtual_move.rs)

- Just range-check each (16-bit) chunk of the operand to be at most 16-bit (instead of an arbitrary field element)

- Used in `DIV`, `DIVU`, `REM`, and `REMU`

29. [`VirtualMOVSIGN`](../../../jolt-core/src/jolt/instruction/virtual_movsign.rs)

- Returns (max `u32/u64`) if the first operand's sign bit is 1, and 0 otherwise

- Used in `MULH` and `MULHSU`

### Sequence of Virtual Instructions

See [M extension](../spec/m_extension.md) for the description of the virtual instructions (MULH, MULHSU, DIV, DIVU, REM, REMU).

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