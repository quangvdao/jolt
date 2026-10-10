//! RV64I encodings from the RISC-V unprivileged ISA manual.
//! Registers must be in `0..32`. Signed immediates use their encoded units:
//! U-type values count 4096-byte units; branch and jump offsets count bytes.
//! Operand bounds are checked with assertions, since violations are test bugs.

fn reg(value: u8) -> u32 {
    assert!(value < 32, "register x{value} is outside x0..x31");
    u32::from(value)
}

fn signed(value: i32, bits: u32) -> u32 {
    let limit = 1_i32 << (bits - 1);
    assert!(
        (-limit..limit).contains(&value),
        "immediate {value} does not fit a signed {bits}-bit field"
    );
    (value as u32) & ((1_u32 << bits) - 1)
}

fn i_type(opcode: u32, funct3: u32, rd: u8, rs1: u8, imm: i32) -> u32 {
    (signed(imm, 12) << 20) | (reg(rs1) << 15) | (funct3 << 12) | (reg(rd) << 7) | opcode
}

fn r_type(opcode: u32, funct3: u32, funct7: u32, rd: u8, rs1: u8, rs2: u8) -> u32 {
    (funct7 << 25) | (reg(rs2) << 20) | (reg(rs1) << 15) | (funct3 << 12) | (reg(rd) << 7) | opcode
}

fn branch(funct3: u32, rs1: u8, rs2: u8, offset: i32) -> u32 {
    assert!(offset % 2 == 0, "branch offset {offset} is not even");
    let imm = signed(offset, 13);
    ((imm >> 12) << 31)
        | (((imm >> 5) & 0x3f) << 25)
        | (reg(rs2) << 20)
        | (reg(rs1) << 15)
        | (funct3 << 12)
        | (((imm >> 1) & 0xf) << 8)
        | (((imm >> 11) & 1) << 7)
        | 0x63
}

fn store(funct3: u32, rs1: u8, rs2: u8, imm: i32) -> u32 {
    let imm = signed(imm, 12);
    ((imm >> 5) << 25)
        | (reg(rs2) << 20)
        | (reg(rs1) << 15)
        | (funct3 << 12)
        | ((imm & 0x1f) << 7)
        | 0x23
}

fn shift(opcode: u32, funct3: u32, arithmetic: bool, rd: u8, rs1: u8, shamt: i32) -> u32 {
    let limit = if opcode == 0x1b { 32 } else { 64 };
    assert!(
        (0..limit).contains(&shamt),
        "shift amount {shamt} is outside 0..{limit}"
    );
    i_type(
        opcode,
        funct3,
        rd,
        rs1,
        shamt | if arithmetic { 0x400 } else { 0 },
    )
}

/// Encode LUI with a signed 20-bit immediate in 4096-byte units.
pub fn lui(rd: u8, imm: i32) -> u32 {
    (signed(imm, 20) << 12) | (reg(rd) << 7) | 0x37
}

/// Encode AUIPC with a signed 20-bit immediate in 4096-byte units.
pub fn auipc(rd: u8, imm: i32) -> u32 {
    (signed(imm, 20) << 12) | (reg(rd) << 7) | 0x17
}

/// Encode JAL with an even signed 21-bit byte offset.
pub fn jal(rd: u8, offset: i32) -> u32 {
    assert!(offset % 2 == 0, "jump offset {offset} is not even");
    let imm = signed(offset, 21);
    ((imm >> 20) << 31)
        | (((imm >> 1) & 0x3ff) << 21)
        | (((imm >> 11) & 1) << 20)
        | (((imm >> 12) & 0xff) << 12)
        | (reg(rd) << 7)
        | 0x6f
}

/// Encode JALR with a signed 12-bit byte displacement.
pub fn jalr(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x67, 0, rd, rs1, imm)
}

/// Encode BEQ with an even signed 13-bit byte offset.
pub fn beq(rs1: u8, rs2: u8, offset: i32) -> u32 {
    branch(0, rs1, rs2, offset)
}

/// Encode BNE with an even signed 13-bit byte offset.
pub fn bne(rs1: u8, rs2: u8, offset: i32) -> u32 {
    branch(1, rs1, rs2, offset)
}

/// Encode BLT with an even signed 13-bit byte offset.
pub fn blt(rs1: u8, rs2: u8, offset: i32) -> u32 {
    branch(4, rs1, rs2, offset)
}

/// Encode BGE with an even signed 13-bit byte offset.
pub fn bge(rs1: u8, rs2: u8, offset: i32) -> u32 {
    branch(5, rs1, rs2, offset)
}

/// Encode BLTU with an even signed 13-bit byte offset.
pub fn bltu(rs1: u8, rs2: u8, offset: i32) -> u32 {
    branch(6, rs1, rs2, offset)
}

/// Encode BGEU with an even signed 13-bit byte offset.
pub fn bgeu(rs1: u8, rs2: u8, offset: i32) -> u32 {
    branch(7, rs1, rs2, offset)
}

/// Encode LB with a signed 12-bit byte displacement.
pub fn lb(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x03, 0, rd, rs1, imm)
}

/// Encode LH with a signed 12-bit byte displacement.
pub fn lh(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x03, 1, rd, rs1, imm)
}

/// Encode LW with a signed 12-bit byte displacement.
pub fn lw(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x03, 2, rd, rs1, imm)
}

/// Encode LBU with a signed 12-bit byte displacement.
pub fn lbu(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x03, 4, rd, rs1, imm)
}

/// Encode LHU with a signed 12-bit byte displacement.
pub fn lhu(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x03, 5, rd, rs1, imm)
}

/// Encode LWU with a signed 12-bit byte displacement.
pub fn lwu(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x03, 6, rd, rs1, imm)
}

/// Encode LD with a signed 12-bit byte displacement.
pub fn ld(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x03, 3, rd, rs1, imm)
}

/// Encode SB storing `rs2`, with a signed 12-bit byte displacement from `rs1`.
pub fn sb(rs1: u8, rs2: u8, imm: i32) -> u32 {
    store(0, rs1, rs2, imm)
}

/// Encode SH storing `rs2`, with a signed 12-bit byte displacement from `rs1`.
pub fn sh(rs1: u8, rs2: u8, imm: i32) -> u32 {
    store(1, rs1, rs2, imm)
}

/// Encode SW storing `rs2`, with a signed 12-bit byte displacement from `rs1`.
pub fn sw(rs1: u8, rs2: u8, imm: i32) -> u32 {
    store(2, rs1, rs2, imm)
}

/// Encode SD storing `rs2`, with a signed 12-bit byte displacement from `rs1`.
pub fn sd(rs1: u8, rs2: u8, imm: i32) -> u32 {
    store(3, rs1, rs2, imm)
}

/// Encode ADDI with a signed 12-bit immediate.
pub fn addi(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x13, 0, rd, rs1, imm)
}

/// Encode SLTI with a signed 12-bit immediate.
pub fn slti(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x13, 2, rd, rs1, imm)
}

/// Encode SLTIU with a signed 12-bit immediate.
pub fn sltiu(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x13, 3, rd, rs1, imm)
}

/// Encode XORI with a signed 12-bit immediate.
pub fn xori(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x13, 4, rd, rs1, imm)
}

/// Encode ORI with a signed 12-bit immediate.
pub fn ori(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x13, 6, rd, rs1, imm)
}

/// Encode ANDI with a signed 12-bit immediate.
pub fn andi(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x13, 7, rd, rs1, imm)
}

/// Encode SLLI with a shift amount in `0..64`.
pub fn slli(rd: u8, rs1: u8, shamt: i32) -> u32 {
    shift(0x13, 1, false, rd, rs1, shamt)
}

/// Encode SRLI with a shift amount in `0..64`.
pub fn srli(rd: u8, rs1: u8, shamt: i32) -> u32 {
    shift(0x13, 5, false, rd, rs1, shamt)
}

/// Encode SRAI with a shift amount in `0..64`.
pub fn srai(rd: u8, rs1: u8, shamt: i32) -> u32 {
    shift(0x13, 5, true, rd, rs1, shamt)
}

/// Encode ADD; every register operand must be in `0..32`.
pub fn add(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 0, 0x00, rd, rs1, rs2)
}

/// Encode SUB; every register operand must be in `0..32`.
pub fn sub(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 0, 0x20, rd, rs1, rs2)
}

/// Encode SLL; every register operand must be in `0..32`.
pub fn sll(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 1, 0x00, rd, rs1, rs2)
}

/// Encode SLT; every register operand must be in `0..32`.
pub fn slt(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 2, 0x00, rd, rs1, rs2)
}

/// Encode SLTU; every register operand must be in `0..32`.
pub fn sltu(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 3, 0x00, rd, rs1, rs2)
}

/// Encode XOR; every register operand must be in `0..32`.
pub fn xor(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 4, 0x00, rd, rs1, rs2)
}

/// Encode SRL; every register operand must be in `0..32`.
pub fn srl(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 5, 0x00, rd, rs1, rs2)
}

/// Encode SRA; every register operand must be in `0..32`.
pub fn sra(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 5, 0x20, rd, rs1, rs2)
}

/// Encode OR; every register operand must be in `0..32`.
pub fn or(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 6, 0x00, rd, rs1, rs2)
}

/// Encode AND; every register operand must be in `0..32`.
pub fn and(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x33, 7, 0x00, rd, rs1, rs2)
}

/// Encode ADDIW with a signed 12-bit immediate.
pub fn addiw(rd: u8, rs1: u8, imm: i32) -> u32 {
    i_type(0x1b, 0, rd, rs1, imm)
}

/// Encode SLLIW with a shift amount in `0..32`.
pub fn slliw(rd: u8, rs1: u8, shamt: i32) -> u32 {
    shift(0x1b, 1, false, rd, rs1, shamt)
}

/// Encode SRLIW with a shift amount in `0..32`.
pub fn srliw(rd: u8, rs1: u8, shamt: i32) -> u32 {
    shift(0x1b, 5, false, rd, rs1, shamt)
}

/// Encode SRAIW with a shift amount in `0..32`.
pub fn sraiw(rd: u8, rs1: u8, shamt: i32) -> u32 {
    shift(0x1b, 5, true, rd, rs1, shamt)
}

/// Encode ADDW; every register operand must be in `0..32`.
pub fn addw(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x3b, 0, 0x00, rd, rs1, rs2)
}

/// Encode SUBW; every register operand must be in `0..32`.
pub fn subw(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x3b, 0, 0x20, rd, rs1, rs2)
}

/// Encode SLLW; every register operand must be in `0..32`.
pub fn sllw(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x3b, 1, 0x00, rd, rs1, rs2)
}

/// Encode SRLW; every register operand must be in `0..32`.
pub fn srlw(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x3b, 5, 0x00, rd, rs1, rs2)
}

/// Encode SRAW; every register operand must be in `0..32`.
pub fn sraw(rd: u8, rs1: u8, rs2: u8) -> u32 {
    r_type(0x3b, 5, 0x20, rd, rs1, rs2)
}

/// Encode FENCE with four-bit predecessor and successor masks and `fm = 0`.
pub fn fence(pred: u8, succ: u8) -> u32 {
    assert!(pred < 16, "FENCE predecessor mask {pred} is outside 0..16");
    assert!(succ < 16, "FENCE successor mask {succ} is outside 0..16");
    (u32::from(pred) << 24) | (u32::from(succ) << 20) | 0x0f
}

/// Encode ECALL with the reserved register and immediate fields clear.
pub fn ecall() -> u32 {
    0x0000_0073
}

/// Encode EBREAK with the reserved register fields clear.
pub fn ebreak() -> u32 {
    0x0010_0073
}
