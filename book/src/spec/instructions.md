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

- $\verb|WORD_SIZE|$ is the number of bits in each operand.

- $C$ is the number of chunks that each operand is split into.

- $M$ is the size of the subtables (so that $m := \log_2(M) / 2$ is the number of bits in each chunk).

In the codebase, we assert that: $\verb|WORD_SIZE| \leq C \cdot \log_2(M)$. This guarantees that all the bits in each operand are divided into chunks (though the last chunk might be shorter).

- 

## Instructions List

### Logical & Arithmetic Instructions

AND / OR / XOR

- Perform bitwise operations using the corresponding subtables

ADD / SUB / MUL / MULU / MULHU

- Perform the operation in-circuit, and then range-check the result to be `u32/u64`

SLL / SRL / SRA

- 

### Set & Branch Instructions

SLT / SLTU / BEQ / BNE / BGE / BGEU

- 

### Load & Store Instructions

LB / LH / SB / SH / SW

(why no LW?)

SW

- Returns the lower 32 bits of the (second) operand

- Is constrained via two `IdentitySubtable`s on the third and fourth chunks of the second operand (for 32 bits only, what about 64?)

- What about the other chunks? Why can they be unconstrained? The answer is that the other chunks are simply ignored. There's no subtable lookup for them, so they are not present when combining lookups.


### Virtual Instructions

VirtualAdvice

- Only range-check the advice to be `u32/u64`

VirtualAssertLTE

- Compute $x \le y$ as a combination of strict less-than and equality

VirtualAssertValidDiv0

- Inputs $x, y$ are interpreted as (unsigned) divisor and quotient
- Output 1 if $x \ne 0$ or if $x = 0$ and $y = 2 ^ \verb|WORD_SIZE| - 1$

VirtualAssertValidSignedRemainder

- 

VirtualAssertValidUnsignedRemainder

- 

VirtualMove

- Just range-check each (16-bit) chunk of the operand to be at most 16-bit (instead of an arbitrary field element)

VirtualMOVSIGN

- Returns (max `u32/u64`) if the first operand's sign bit is 1, and 0 otherwise

### Sequence of Virtual Instructions

MULH / MULHSU / DIV / DIVU / REM / REMU

NOTE: some instructions are actually sequences of other (virtual) instructions, which is the case for the `MUL` and `DIV` instructions.

1. `ADDInstruction(x, y)`

- Expected output: $x + y$, truncated to $\verb|WORD_SIZE|$ bits.

- Subtables: `TruncateOverflowSubtable`, `IdentitySubtable`

- How to query subtables:

- How to combine lookup outputs:

2. `SUBInstruction(x, y)`

- Expected output: $x - y$, truncated to $\verb|WORD_SIZE|$ bits.

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

3. `ANDInstruction(x, y)`

- Expected output: $x \land y$ (performed bitwise on $\verb|WORD_SIZE|$ bit operands).

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

1. `ORInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

5. `XORInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

6. `LBInstruction(x)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

7. `LHInstruction(x)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

8. `SBInstruction(x)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

9. `SHInstruction(x)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

10. `SWInstruction(x)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

11. `BEQInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

12. `BGEInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

13. `BGEUInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

14. `BNEInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

15. `SLTInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

16. `SLTUInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

17. `SLLInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

18. `SRAInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

19. `SRLInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

20. `MOVSIGNInstruction(x)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

21. `MULInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

22. `MULUInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

23. `MULHUInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

24. `ADVICEInstruction(x)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

25. `ASSERTLTEInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

26. `AssertValidSignedRemainderInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

27. `AssertValidUnsignedRemainderInstruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs:

28. `AssertValidDiv0Instruction(x, y)`

- Subtables: 

- How to query subtables:

- How to combine lookup outputs: