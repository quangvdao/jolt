# Specification of Jolt's R1CS Constraints

## Inputs & Witness

Constants:

- `RAM_OPS_PER_INSTRUCTION = 4`
- `PC_START_ADDRESS = 0x80000000`
- `PC_NOOP_SHIFT = 0x00000000`
- `IMM_TWOS_COMPLEMENT_OFFSET = 0xffffffff + 1`

Bytecode part is public, and consists of the following variables:

- `Bytecode_A`
- `Bytecode_ELFAddress`
- `Bytecode_Bitflags`
- `Bytecode_RS1`
- `Bytecode_RS2`
- `Bytecode_RD`
- `Bytecode_Imm`

Other parts of the witness includes:

1. Read/written values
   - `RAM_A`
   - `RS1_Read`
   - `RS2_Read`
   - `RD_Read`
   - `RAM_Read(i)` (for `i = 0` to `RAM_OPS_PER_INSTRUCTION - 1`)
   - `RD_Write`
   - `RAM_Write(i)` (for `i = 0` to `RAM_OPS_PER_INSTRUCTION - 1`)

2. Chunks for lookups and lookup output
   - `ChunksQuery(i)` (for `i = 0` to `C - 1`)
   - `LookupOutput`
   - `ChunksX(i)` (for `i = 0` to `C - 1`)
   - `ChunksY(i)` (for `i = 0` to `C - 1`)

3. Circuit flags, instruction flags
   - `OpFlags(CircuitFlags)`: denote the circuit flags
     - `LeftOperandIsPC`
     - `RightOperandIsImm`
     - `Load`
     - `Store`
     - `Jump`
     - `Branch`
     - `WriteLookupOutputToRD`
     - `ImmSignBit`
     - `ConcatLookupQueryChunks`
     - `Virtual`
     - `Assert`
     - `DoNotUpdatePC`
   - `InstructionFlags(RV32I)`: denote the instruction in `RV32I` enum (see [Instructions Specification](../spec/instructions.md))

4. Auxiliary variables
   - `LeftLookupOperand`
   - `RightLookupOperand`
   - `ImmSigned`
   - `Product`
   - `RelevantYChunk(i)` (for `i = 0` to `C - 1`)
   - `WriteLookupOutputToRD`
   - `WritePCtoRD`
   - `NextPCJump`
   - `ShouldBranch`
   - `NextPC`

## R1CS Constraints

We will write the name of the R1CS variable instead of the variable itself.

1. For each instruction flags and circuit flags, constrain them to be binary:

   $$c \cdot (c - 1) = 0 \qquad\text{ for all }c \in \{\mathsf{InstructionFlags},\mathsf{CircuitFlags}}.$$

2. Constrain that the bytecode bitflags are a packed encoding of the instruction flags and circuit flags:

   $$\sum_{i=0}^{m-1} 2^{m - i - 1} \cdot c_i = \sum_{i=0}^{m-1} 2^{m - i - 1} \cdot i_i \qquad\text{ where } c_i, i_i \in \{0,1\}.$$

3. Set the left lookup operand to be the program counter if the circuit flag `LeftOperandIsPC` is set, or the value of the first register otherwise:

   $$\mathsf{LeftLookupOperand} = \mathsf{if} \left( \mathsf{LeftOperandIsPC}, 4 \cdot \mathsf{Bytecode\_ELFAddress} + \mathsf{PC\_StartAddress} - \mathsf{PC\_NoopShift}, \mathsf{RS1\_Read} \right).$$

4. Set the right lookup operand to be the immediate `Bytecode_Imm` if the circuit flag `RightOperandIsImm` is set, or the value of the second register otherwise:

   $$\mathsf{RightLookupOperand} = \mathsf{if} \left( \mathsf{RightOperandIsImm}, \mathsf{Bytecode\_Imm}, \mathsf{RS2\_Read} \right).$$

5. Convert the immediate `Bytecode_Imm` to its two's complement representation if the circuit flag `ImmSignBit` is set, otherwise leave it as is:

   $$\mathsf{imm\_signed} = \mathsf{if} \left( \mathsf{imm\_sign\_bit}, \mathsf{bytecode\_imm} - \mathsf{imm\_twos\_complement\_offset}, \mathsf{bytecode\_imm} \right).$$

6. If the circuit flag `Load` or `Store` is set, constrain the read value to be equal to the write value:

   $$\mathsf{if} \left( \mathsf{load} + \mathsf{store}, \mathsf{rd\_read} = \mathsf{rd\_write}, 1 \right).$$

## Intuition for Correctness

