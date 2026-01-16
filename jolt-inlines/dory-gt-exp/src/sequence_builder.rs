//! Host-side inline sequence builder for `BN254_GT_EXP`.
//!
//! SECURITY NOTE:
//! This file is security-critical. We implement arithmetic over BN254 Fq (and, later, extensions)
//! directly as RV64 instruction sequences. Any bug here can silently break cryptographic
//! correctness. Prefer small, well-tested building blocks.
//!
//! Current staged implementation plan (see `BN254_GT_EXP_INLINE_DESIGN.md`):
//! - Step 1: implement BN254 base field (Fq) Montgomery arithmetic helpers + tests
//! - Step 2: build Fq2/Fq6/Fq12 ops + tests
//! - Step 3: wire into `BN254_GT_EXP` (starting with exp==2 => square) + differential tests

#![allow(dead_code)]
#![allow(clippy::identity_op)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use core::array;

use tracer::instruction::{
    add::ADD, addi::ADDI, format::format_inline::FormatInline, ld::LD, mul::MUL, mulhu::MULHU,
    sd::SD, sltu::SLTU, srli::SRLI, sub::SUB, virtual_assert_eq::VirtualAssertEQ, Instruction,
};
use tracer::utils::{inline_helpers::InstrAssembler, virtual_registers::VirtualRegisterGuard};

#[cfg(feature = "host")]
use ark_ff::MontConfig;

// -----------------------------
// BN254 base field constants
// -----------------------------
//
// BN254 base field (Fq) is `Fp256<MontBackend<FqConfig, 4>>`, i.e. 4x u64 limbs in Montgomery form.
// We pull modulus and INV from the arkworks config to guarantee we match the upstream semantics.
#[cfg(feature = "host")]
const BN254_FQ_MODULUS: [u64; 4] = <ark_bn254::FqConfig as MontConfig<4>>::MODULUS.0;
#[cfg(feature = "host")]
const BN254_FQ_INV: u64 = <ark_bn254::FqConfig as MontConfig<4>>::INV;
#[cfg(feature = "host")]
const BN254_FQ_ONE: [u64; 4] = <ark_bn254::FqConfig as MontConfig<4>>::R.0;

// -----------------------------
// Small helpers (64-bit limb arithmetic)
// -----------------------------

/// Helper for a 256-bit BN254 Fq element in registers (4 little-endian u64 limbs).
type FqRegs = [u8; 4];

#[inline(always)]
fn fq_regs(vr: &[VirtualRegisterGuard], start: usize) -> FqRegs {
    [
        *vr[start + 0],
        *vr[start + 1],
        *vr[start + 2],
        *vr[start + 3],
    ]
}

/// Emits `dst = src` (register copy).
#[inline(always)]
fn mov(asm: &mut InstrAssembler, dst: u8, src: u8) {
    asm.emit_i::<ADDI>(dst, src, 0);
}

/// Emits `dst = imm` using `ADDI dst, x0, imm`.
#[inline(always)]
fn li(asm: &mut InstrAssembler, dst: u8, imm: u64) {
    asm.emit_i::<ADDI>(dst, 0, imm);
}

#[inline(always)]
fn load_fq(asm: &mut InstrAssembler, dst: FqRegs, base: u8, offset_bytes: i64) {
    for i in 0..4 {
        asm.emit_ld::<LD>(dst[i], base, offset_bytes + (i as i64) * 8);
    }
}

#[inline(always)]
fn store_fq(asm: &mut InstrAssembler, base: u8, offset_bytes: i64, src: FqRegs) {
    for i in 0..4 {
        asm.emit_s::<SD>(base, src[i], offset_bytes + (i as i64) * 8);
    }
}

/// Adds two u64 registers: `dst = a + b` and sets `carry = (dst < a_old)` (unsigned).
///
/// IMPORTANT: This works even when `dst == a` by saving `a` into `tmp`.
#[inline(always)]
fn add_u64_set_carry(asm: &mut InstrAssembler, dst: u8, a: u8, b: u8, carry: u8, tmp: u8) {
    mov(asm, tmp, a); // tmp = a_old
    asm.emit_r::<ADD>(dst, a, b);
    // carry = (dst < a_old) ? 1 : 0
    asm.emit_r::<SLTU>(carry, dst, tmp);
}

/// Subtracts two u64 registers: `dst = a - b` and sets `borrow = (a_old < b)` (unsigned).
///
/// IMPORTANT: This works even when `dst == a` by saving `a` into `tmp`.
#[inline(always)]
fn sub_u64_set_borrow(asm: &mut InstrAssembler, dst: u8, a: u8, b: u8, borrow: u8, tmp: u8) {
    mov(asm, tmp, a); // tmp = a_old
    asm.emit_r::<SUB>(dst, a, b);
    asm.emit_r::<SLTU>(borrow, tmp, b);
}

/// Multiplies two u64 registers, producing `(lo, hi)` into provided registers.
#[inline(always)]
fn mul_u64_wide(asm: &mut InstrAssembler, lo: u8, hi: u8, a: u8, b: u8) {
    asm.emit_r::<MULHU>(hi, a, b);
    asm.emit_r::<MUL>(lo, a, b);
}

// -----------------------------
// BN254 Fq Montgomery arithmetic (register-level)
// -----------------------------

/// Fixed scratch register layout for BN254 Fq ops.
///
/// We keep this small and *reused* to avoid allocating many virtual registers, which would add
/// cleanup cost at `finalize_inline()` time.
struct FqScratch {
    // Constants
    inv: u8,
    mod_limb: FqRegs,
    // Temps used by `mac_*` and carry handling
    t0: u8,
    t1: u8,
    t2: u8,
    t3: u8,
    // Wide multiply temps
    mul_lo: u8,
    mul_hi: u8,
    // Carry/borrow temps (u64 values, often 0/1)
    c0: u8,
    c1: u8,
    c2: u8,
}

impl FqScratch {
    fn new(vr: &[VirtualRegisterGuard]) -> Self {
        // NOTE: The caller must allocate enough regs; indices are fixed by convention here.
        // Layout:
        // 0..4   : inv + modulus limbs
        // 5..    : temporaries
        Self {
            inv: *vr[0],
            mod_limb: [*vr[1], *vr[2], *vr[3], *vr[4]],
            t0: *vr[5],
            t1: *vr[6],
            t2: *vr[7],
            t3: *vr[8],
            mul_lo: *vr[9],
            mul_hi: *vr[10],
            c0: *vr[11],
            c1: *vr[12],
            c2: *vr[13],
        }
    }

    fn init_constants(&self, asm: &mut InstrAssembler) {
        li(asm, self.inv, BN254_FQ_INV);
        li(asm, self.mod_limb[0], BN254_FQ_MODULUS[0]);
        li(asm, self.mod_limb[1], BN254_FQ_MODULUS[1]);
        li(asm, self.mod_limb[2], BN254_FQ_MODULUS[2]);
        li(asm, self.mod_limb[3], BN254_FQ_MODULUS[3]);
    }
}

/// `mac(a, b, c)`:
/// Computes `a + b*c`, returns low 64 bits in `out`, and writes high 64 bits to `carry`.
#[inline(always)]
fn fq_mac(asm: &mut InstrAssembler, out: u8, a: u8, b: u8, c: u8, carry: u8, s: &FqScratch) {
    mul_u64_wide(asm, s.mul_lo, s.mul_hi, b, c);
    add_u64_set_carry(asm, out, a, s.mul_lo, s.c0, s.t3); // c0 is carry bit (0/1)
                                                          // carry = mul_hi + c0
    asm.emit_r::<ADD>(carry, s.mul_hi, s.c0);
}

/// `mac_discard(a, b, c)`:
/// Computes `a + b*c`, discards low 64 bits, writes high 64 bits to `carry`.
#[inline(always)]
fn fq_mac_discard(asm: &mut InstrAssembler, a: u8, b: u8, c: u8, carry: u8, s: &FqScratch) {
    mul_u64_wide(asm, s.mul_lo, s.mul_hi, b, c);
    // tmp = a + mul_lo; c0 = carry bit
    // IMPORTANT: do not clobber `t0` here; the caller often uses `t0` to hold `k`.
    add_u64_set_carry(asm, s.t3, a, s.mul_lo, s.c0, s.c2);
    // carry = mul_hi + c0
    asm.emit_r::<ADD>(carry, s.mul_hi, s.c0);
}

/// `mac_with_carry(a, b, c, carry_in)`:
/// Computes `a + b*c + carry_in`, returns low 64 bits in `out`, writes high 64 bits to `carry_io`.
#[inline(always)]
fn fq_mac_with_carry(
    asm: &mut InstrAssembler,
    out: u8,
    a: u8,
    b: u8,
    c: u8,
    carry_io: u8,
    s: &FqScratch,
) {
    mul_u64_wide(asm, s.mul_lo, s.mul_hi, b, c);
    // out = a + mul_lo; bit0 = carry bit
    add_u64_set_carry(asm, out, a, s.mul_lo, s.c0, s.t3);
    // Save out_old (after first add) into t3 (we no longer need a_old).
    mov(asm, s.t3, out);
    // tmp = carry_in (must not clobber carry_io itself)
    mov(asm, s.c2, carry_io);
    // out = out + carry_in; bit1 = carry bit (result < out_old)
    asm.emit_r::<ADD>(out, out, s.c2);
    asm.emit_r::<SLTU>(s.c1, out, s.t3);
    // carry_io = mul_hi + bit0 + bit1
    asm.emit_r::<ADD>(carry_io, s.mul_hi, s.c0);
    asm.emit_r::<ADD>(carry_io, carry_io, s.c1);
}

/// Conditional select: `dst = (mask == 0) ? a : b`, where mask is either 0 or all-ones.
#[inline(always)]
fn select_u64_masked(asm: &mut InstrAssembler, dst: u8, a: u8, b: u8, mask: u8, s: &FqScratch) {
    // dst = a ^ ((a ^ b) & mask)
    asm.emit_r::<SUB>(s.t0, a, 0); // t0 = a (copy via SUB with x0)
    asm.emit_r::<SUB>(s.t1, b, 0); // t1 = b
    asm.emit_r::<SUB>(dst, a, 0); // dst = a
                                  // t2 = a ^ b
                                  // We don't have XOR imported; avoid expanding scope until needed.
                                  // For now, use arithmetic select below in the Fq-level functions.
                                  // (Kept here as a placeholder for future constant-time selects.)
    let _ = (s.t0, s.t1, mask);
}

/// Compute `out = a + b (mod p)` for BN254 Fq limbs.
///
/// This assumes `a,b` are canonical (< p). Output is canonical.
fn fq_add_mod(asm: &mut InstrAssembler, out: FqRegs, a: FqRegs, b: FqRegs, s: &FqScratch) {
    // 1) out = a + b (multi-limb)
    li(asm, s.c2, 0); // carry (0/1)
    for i in 0..4 {
        // out[i] = a[i] + b[i]
        add_u64_set_carry(asm, out[i], a[i], b[i], s.c0, s.t3); // c0 = carry1
                                                                // out[i] = out[i] + carry_in
        mov(asm, s.t3, out[i]); // save out_old
        asm.emit_r::<ADD>(out[i], out[i], s.c2);
        asm.emit_r::<SLTU>(s.c1, out[i], s.t3); // carry2
                                                // carry_out = carry1 + carry2
        asm.emit_r::<ADD>(s.c2, s.c0, s.c1);
    }

    // 2) Reduce: out = out - p; if borrow==1 add p back.
    li(asm, s.t2, 0); // borrow (0/1)
    for i in 0..4 {
        sub_u64_set_borrow(asm, s.t3, out[i], s.mod_limb[i], s.c0, s.c2); // c0 = borrow1
        asm.emit_r::<SUB>(out[i], s.t3, s.t2);
        asm.emit_r::<SLTU>(s.c1, s.t3, s.t2); // borrow2
        asm.emit_r::<ADD>(s.t2, s.c0, s.c1); // borrow_out
    }
    li(asm, s.c2, 0); // carry (0/1)
    for i in 0..4 {
        asm.emit_r::<MUL>(s.t3, s.t2, s.mod_limb[i]); // t3 = borrow * mod[i]
        add_u64_set_carry(asm, out[i], out[i], s.t3, s.c0, s.t1);
        mov(asm, s.t1, out[i]);
        asm.emit_r::<ADD>(out[i], out[i], s.c2);
        asm.emit_r::<SLTU>(s.c1, out[i], s.t1); // carry2 (from adding carry_in)
        asm.emit_r::<ADD>(s.c2, s.c0, s.c1); // carry_out
    }
}

/// Compute `out = a - b (mod p)` for BN254 Fq limbs.
///
/// This assumes `a,b` are canonical (< p). Output is canonical.
fn fq_sub_mod(asm: &mut InstrAssembler, out: FqRegs, a: FqRegs, b: FqRegs, s: &FqScratch) {
    // out = a - b; if borrow==1 add p back.
    li(asm, s.t2, 0); // borrow (0/1)
    for i in 0..4 {
        sub_u64_set_borrow(asm, s.t3, a[i], b[i], s.c0, s.c2); // c0 = borrow1
        asm.emit_r::<SUB>(out[i], s.t3, s.t2);
        asm.emit_r::<SLTU>(s.c1, s.t3, s.t2); // borrow2
        asm.emit_r::<ADD>(s.t2, s.c0, s.c1);
    }
    // Add modulus back if borrow==1.
    li(asm, s.c2, 0); // carry
    for i in 0..4 {
        asm.emit_r::<MUL>(s.t3, s.t2, s.mod_limb[i]); // t3 = borrow * mod[i]
        add_u64_set_carry(asm, out[i], out[i], s.t3, s.c0, s.t1);
        mov(asm, s.t1, out[i]);
        asm.emit_r::<ADD>(out[i], out[i], s.c2);
        asm.emit_r::<SLTU>(s.c1, out[i], s.t1);
        asm.emit_r::<ADD>(s.c2, s.c0, s.c1);
    }
}

/// Compute `out = -a (mod p)` for BN254 Fq limbs.
///
/// This assumes `a` is canonical (< p). Output is canonical.
fn fq_neg_mod(asm: &mut InstrAssembler, out: FqRegs, a: FqRegs, s: &FqScratch) {
    // out = p - a
    li(asm, s.t2, 0); // borrow (0/1)
    for i in 0..4 {
        sub_u64_set_borrow(asm, s.t3, s.mod_limb[i], a[i], s.c0, s.c2);
        asm.emit_r::<SUB>(out[i], s.t3, s.t2);
        asm.emit_r::<SLTU>(s.c1, s.t3, s.t2);
        asm.emit_r::<ADD>(s.t2, s.c0, s.c1);
    }
    // Conditional subtract modulus to map p -> 0 when a==0.
    li(asm, s.t2, 0); // borrow (0/1)
    for i in 0..4 {
        sub_u64_set_borrow(asm, s.t3, out[i], s.mod_limb[i], s.c0, s.c2);
        asm.emit_r::<SUB>(out[i], s.t3, s.t2);
        asm.emit_r::<SLTU>(s.c1, s.t3, s.t2);
        asm.emit_r::<ADD>(s.t2, s.c0, s.c1);
    }
    // Add modulus back if we underflowed (i.e., out < p).
    li(asm, s.c2, 0); // carry
    for i in 0..4 {
        asm.emit_r::<MUL>(s.t3, s.t2, s.mod_limb[i]);
        add_u64_set_carry(asm, out[i], out[i], s.t3, s.c0, s.t1);
        mov(asm, s.t1, out[i]);
        asm.emit_r::<ADD>(out[i], out[i], s.c2);
        asm.emit_r::<SLTU>(s.c1, out[i], s.t1);
        asm.emit_r::<ADD>(s.c2, s.c0, s.c1);
    }
}

// -----------------------------
// BN254 Fq2 arithmetic (register-level)
// -----------------------------

/// Helper for an Fq2 element in registers: (c0: Fq, c1: Fq).
#[derive(Clone, Copy)]
struct Fq2Regs {
    c0: FqRegs,
    c1: FqRegs,
}

#[inline(always)]
fn copy_fq(asm: &mut InstrAssembler, dst: FqRegs, src: FqRegs) {
    for i in 0..4 {
        mov(asm, dst[i], src[i]);
    }
}

#[inline(always)]
fn copy_fq2(asm: &mut InstrAssembler, dst: Fq2Regs, src: Fq2Regs) {
    copy_fq(asm, dst.c0, src.c0);
    copy_fq(asm, dst.c1, src.c1);
}

#[inline(always)]
fn load_fq2(asm: &mut InstrAssembler, dst: Fq2Regs, base: u8, offset_bytes: i64) {
    load_fq(asm, dst.c0, base, offset_bytes);
    load_fq(asm, dst.c1, base, offset_bytes + 32);
}

#[inline(always)]
fn store_fq2(asm: &mut InstrAssembler, base: u8, offset_bytes: i64, src: Fq2Regs) {
    store_fq(asm, base, offset_bytes, src.c0);
    store_fq(asm, base, offset_bytes + 32, src.c1);
}

fn fq2_add_mod(asm: &mut InstrAssembler, out: Fq2Regs, a: Fq2Regs, b: Fq2Regs, s: &FqScratch) {
    fq_add_mod(asm, out.c0, a.c0, b.c0, s);
    fq_add_mod(asm, out.c1, a.c1, b.c1, s);
}

fn fq2_sub_mod(asm: &mut InstrAssembler, out: Fq2Regs, a: Fq2Regs, b: Fq2Regs, s: &FqScratch) {
    fq_sub_mod(asm, out.c0, a.c0, b.c0, s);
    fq_sub_mod(asm, out.c1, a.c1, b.c1, s);
}

fn fq2_double_in_place(asm: &mut InstrAssembler, a: Fq2Regs, s: &FqScratch) {
    fq_add_mod(asm, a.c0, a.c0, a.c0, s);
    fq_add_mod(asm, a.c1, a.c1, a.c1, s);
}

/// Karatsuba multiplication in Fq2 with NONRESIDUE = -1.
///
/// Uses 3 Fq multiplications, and requires three Fq temporaries `t0`, `t1`, `t2`.
fn fq2_mul_karatsuba(
    asm: &mut InstrAssembler,
    out: Fq2Regs,
    a: Fq2Regs,
    b: Fq2Regs,
    t0: FqRegs,
    t1: FqRegs,
    t2: FqRegs,
    s: &FqScratch,
) {
    // v0 = a0*b0 -> out.c0
    fq_mul_mont(asm, out.c0, a.c0, b.c0, s);
    // v1 = a1*b1 -> out.c1
    fq_mul_mont(asm, out.c1, a.c1, b.c1, s);

    // c0 = v0 - v1  (store in out.c0)
    fq_sub_mod(asm, out.c0, out.c0, out.c1, s);

    // t0 = a0 + a1
    fq_add_mod(asm, t0, a.c0, a.c1, s);
    // t1 = b0 + b1
    fq_add_mod(asm, t1, b.c0, b.c1, s);
    // t2 = (a0+a1)*(b0+b1)
    fq_mul_mont(asm, t2, t0, t1, s);

    // t0 = 2*v1
    for i in 0..4 {
        mov(asm, t0[i], out.c1[i]);
    }
    fq_add_mod(asm, t0, t0, t0, s);

    // out.c1 = t2 - c0 - 2*v1
    fq_sub_mod(asm, out.c1, t2, out.c0, s);
    fq_sub_mod(asm, out.c1, out.c1, t0, s);
}

/// Karatsuba multiplication in Fq2 with NONRESIDUE = -1, optimized for register pressure.
///
/// This variant uses **no extra Fq temporaries** beyond `a` and `b` themselves, and therefore
/// intentionally **clobbers** `a.c0`, `a.c1`, and `b.c0` during computation. `b.c1` is left intact.
///
/// Register/aliasing requirements:
/// - `out` must not alias `a` or `b` (because `fq_mul_mont` requires distinct out/input regs).
fn fq2_mul_karatsuba_clobber(
    asm: &mut InstrAssembler,
    out: Fq2Regs,
    a: Fq2Regs,
    b: Fq2Regs,
    s: &FqScratch,
) {
    // v0 = a0*b0 -> out.c0
    fq_mul_mont(asm, out.c0, a.c0, b.c0, s);
    // v1 = a1*b1 -> out.c1
    fq_mul_mont(asm, out.c1, a.c1, b.c1, s);

    // a0 = a0 + a1
    fq_add_mod(asm, a.c0, a.c0, a.c1, s);
    // b0 = b0 + b1
    fq_add_mod(asm, b.c0, b.c0, b.c1, s);

    // a1 = (a0+a1)*(b0+b1)  (store v2 in a1, which is distinct from a0 and b0)
    fq_mul_mont(asm, a.c1, a.c0, b.c0, s);

    // c0 = v0 - v1
    fq_sub_mod(asm, out.c0, out.c0, out.c1, s);

    // b0 = v2 - c0   (reuse b0 as temp)
    fq_sub_mod(asm, b.c0, a.c1, out.c0, s);
    // out.c1 = 2*v1
    fq_add_mod(asm, out.c1, out.c1, out.c1, s);
    // c1 = (v2 - c0) - 2*v1
    fq_sub_mod(asm, out.c1, b.c0, out.c1, s);
}

/// Multiply an Fq2 element by ξ = (9,1) ∈ Fq2 (the BN254 Fq6 nonresidue), in place.
///
/// Implements: (c0 + u*c1) * (9 + u) = (9*c0 - c1) + u*(9*c1 + c0)
/// using only adds/subs/doubles (no Fq multiplications).
fn fq2_mul_by_xi_in_place(
    asm: &mut InstrAssembler,
    fe: Fq2Regs,
    t0: FqRegs,
    t1: FqRegs,
    s: &FqScratch,
) {
    // t0 = 8*c0, t1 = 8*c1
    // t0 = c0; t0 *= 8 via 3 doublings
    for i in 0..4 {
        mov(asm, t0[i], fe.c0[i]);
        mov(asm, t1[i], fe.c1[i]);
    }
    for _ in 0..3 {
        fq_add_mod(asm, t0, t0, t0, s);
        fq_add_mod(asm, t1, t1, t1, s);
    }

    // new_c0 = 9*c0 - c1 = (8*c0 + c0) - c1
    fq_add_mod(asm, t0, t0, fe.c0, s);
    fq_sub_mod(asm, t0, t0, fe.c1, s);

    // new_c1 = 9*c1 + c0 = (8*c1 + c1) + c0
    fq_add_mod(asm, t1, t1, fe.c1, s);
    fq_add_mod(asm, t1, t1, fe.c0, s);

    // write back
    for i in 0..4 {
        mov(asm, fe.c0[i], t0[i]);
        mov(asm, fe.c1[i], t1[i]);
    }
}

// -----------------------------
// BN254 Fq6 arithmetic (memory + register-level)
// -----------------------------

#[derive(Clone, Copy)]
struct Fq6Regs {
    c0: Fq2Regs,
    c1: Fq2Regs,
    c2: Fq2Regs,
}

#[inline(always)]
fn load_fq6(asm: &mut InstrAssembler, dst: Fq6Regs, base: u8, offset_bytes: i64) {
    load_fq2(asm, dst.c0, base, offset_bytes + 0);
    load_fq2(asm, dst.c1, base, offset_bytes + 64);
    load_fq2(asm, dst.c2, base, offset_bytes + 128);
}

#[inline(always)]
fn store_fq6(asm: &mut InstrAssembler, base: u8, offset_bytes: i64, src: Fq6Regs) {
    store_fq2(asm, base, offset_bytes + 0, src.c0);
    store_fq2(asm, base, offset_bytes + 64, src.c1);
    store_fq2(asm, base, offset_bytes + 128, src.c2);
}

fn fq6_add_mod(asm: &mut InstrAssembler, out: Fq6Regs, a: Fq6Regs, b: Fq6Regs, s: &FqScratch) {
    fq2_add_mod(asm, out.c0, a.c0, b.c0, s);
    fq2_add_mod(asm, out.c1, a.c1, b.c1, s);
    fq2_add_mod(asm, out.c2, a.c2, b.c2, s);
}

fn fq6_sub_mod(asm: &mut InstrAssembler, out: Fq6Regs, a: Fq6Regs, b: Fq6Regs, s: &FqScratch) {
    fq2_sub_mod(asm, out.c0, a.c0, b.c0, s);
    fq2_sub_mod(asm, out.c1, a.c1, b.c1, s);
    fq2_sub_mod(asm, out.c2, a.c2, b.c2, s);
}

fn fq6_double_in_place(asm: &mut InstrAssembler, a: Fq6Regs, s: &FqScratch) {
    fq2_double_in_place(asm, a.c0, s);
    fq2_double_in_place(asm, a.c1, s);
    fq2_double_in_place(asm, a.c2, s);
}

/// Multiply an Fq6 element by `v` (the Fq12 quadratic nonresidue), in place:
/// `(c0, c1, c2) := (c2 * ξ, c0, c1)` where ξ = (9,1) ∈ Fq2.
fn fq6_mul_by_nonresidue_v_in_place(
    asm: &mut InstrAssembler,
    fe: Fq6Regs,
    tmp_fq2: Fq2Regs,
    t0: FqRegs,
    t1: FqRegs,
    s: &FqScratch,
) {
    // tmp = c2
    for i in 0..4 {
        mov(asm, tmp_fq2.c0[i], fe.c2.c0[i]);
        mov(asm, tmp_fq2.c1[i], fe.c2.c1[i]);
    }
    // tmp *= ξ
    fq2_mul_by_xi_in_place(asm, tmp_fq2, t0, t1, s);

    // c2 = c1
    for i in 0..4 {
        mov(asm, fe.c2.c0[i], fe.c1.c0[i]);
        mov(asm, fe.c2.c1[i], fe.c1.c1[i]);
    }
    // c1 = c0
    for i in 0..4 {
        mov(asm, fe.c1.c0[i], fe.c0.c0[i]);
        mov(asm, fe.c1.c1[i], fe.c0.c1[i]);
    }
    // c0 = tmp
    for i in 0..4 {
        mov(asm, fe.c0.c0[i], tmp_fq2.c0[i]);
        mov(asm, fe.c0.c1[i], tmp_fq2.c1[i]);
    }
}

/// A small reusable register bundle for Fq2 multiplication and intermediate Fq2 ops.
#[derive(Clone, Copy)]
struct Fq2Work {
    a: Fq2Regs,
    b: Fq2Regs,
    out: Fq2Regs,
    t0: FqRegs,
    t1: FqRegs,
    t2: FqRegs,
}

/// A tighter reusable register bundle for Fq2 ops.
///
/// Used by the exponentiation pipeline to stay under the inline virtual register cap.
#[derive(Clone, Copy)]
struct Fq2WorkTight {
    a: Fq2Regs,
    b: Fq2Regs,
    out: Fq2Regs,
}

#[inline(always)]
fn fq6_add_fq2_into_mem(
    asm: &mut InstrAssembler,
    out_ptr: u8,
    out_off: i64,
    addend: Fq2Regs,
    tmp: Fq2Regs,
    s: &FqScratch,
) {
    load_fq2(asm, tmp, out_ptr, out_off);
    fq2_add_mod(asm, tmp, tmp, addend, s);
    store_fq2(asm, out_ptr, out_off, tmp);
}

#[inline(always)]
fn fq6_sub_fq2_into_mem(
    asm: &mut InstrAssembler,
    out_ptr: u8,
    out_off: i64,
    subtrahend: Fq2Regs,
    tmp: Fq2Regs,
    s: &FqScratch,
) {
    load_fq2(asm, tmp, out_ptr, out_off);
    fq2_sub_mod(asm, tmp, tmp, subtrahend, s);
    store_fq2(asm, out_ptr, out_off, tmp);
}

/// Multiply `lhs` (in regs) by `rhs` (in memory) and write the Fq6 product to memory.
///
/// Uses the schoolbook formula over `Fq2[v]/(v^3 - ξ)` and **does not** use any extra scratch memory
/// beyond the output region itself (i.e., it does not require the 384-byte scratch used by
/// `fq6_mul_karatsuba_mem`).
fn fq6_mul_schoolbook_regmem_to_mem(
    asm: &mut InstrAssembler,
    out_ptr: u8,
    out_off: i64,
    lhs: Fq6Regs,
    rhs_ptr: u8,
    rhs_off: i64,
    w: Fq2WorkTight,
    s: &FqScratch,
) {
    // c0 = a0*b0
    copy_fq2(asm, w.a, lhs.c0);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 0);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    store_fq2(asm, out_ptr, out_off + 0, w.out);

    // c1 = a0*b1
    copy_fq2(asm, w.a, lhs.c0);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 64);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    store_fq2(asm, out_ptr, out_off + 64, w.out);

    // c2 = a0*b2
    copy_fq2(asm, w.a, lhs.c0);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 128);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    store_fq2(asm, out_ptr, out_off + 128, w.out);

    // c1 += a1*b0
    copy_fq2(asm, w.a, lhs.c1);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 0);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 64, w.out, w.a, s);

    // c2 += a1*b1
    copy_fq2(asm, w.a, lhs.c1);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 64);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 128, w.out, w.a, s);

    // c2 += a2*b0
    copy_fq2(asm, w.a, lhs.c2);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 0);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 128, w.out, w.a, s);

    // c0 += ξ*(a1*b2)
    copy_fq2(asm, w.a, lhs.c1);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 128);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq2_mul_by_xi_in_place(asm, w.out, w.a.c0, w.a.c1, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 0, w.out, w.a, s);

    // c0 += ξ*(a2*b1)
    copy_fq2(asm, w.a, lhs.c2);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 64);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq2_mul_by_xi_in_place(asm, w.out, w.a.c0, w.a.c1, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 0, w.out, w.a, s);

    // c1 += ξ*(a2*b2)
    copy_fq2(asm, w.a, lhs.c2);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 128);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq2_mul_by_xi_in_place(asm, w.out, w.a.c0, w.a.c1, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 64, w.out, w.a, s);
}

/// Multiply `lhs` (in regs) by `rhs` (in memory) and **add** the Fq6 product into the existing
/// memory value at `out_ptr + out_off`.
fn fq6_mul_schoolbook_regmem_add_to_mem(
    asm: &mut InstrAssembler,
    out_ptr: u8,
    out_off: i64,
    lhs: Fq6Regs,
    rhs_ptr: u8,
    rhs_off: i64,
    w: Fq2WorkTight,
    s: &FqScratch,
) {
    // c0 += a0*b0
    copy_fq2(asm, w.a, lhs.c0);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 0);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 0, w.out, w.a, s);

    // c1 += a0*b1
    copy_fq2(asm, w.a, lhs.c0);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 64);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 64, w.out, w.a, s);

    // c2 += a0*b2
    copy_fq2(asm, w.a, lhs.c0);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 128);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 128, w.out, w.a, s);

    // c1 += a1*b0
    copy_fq2(asm, w.a, lhs.c1);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 0);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 64, w.out, w.a, s);

    // c2 += a1*b1
    copy_fq2(asm, w.a, lhs.c1);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 64);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 128, w.out, w.a, s);

    // c2 += a2*b0
    copy_fq2(asm, w.a, lhs.c2);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 0);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 128, w.out, w.a, s);

    // c0 += ξ*(a1*b2)
    copy_fq2(asm, w.a, lhs.c1);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 128);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq2_mul_by_xi_in_place(asm, w.out, w.a.c0, w.a.c1, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 0, w.out, w.a, s);

    // c0 += ξ*(a2*b1)
    copy_fq2(asm, w.a, lhs.c2);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 64);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq2_mul_by_xi_in_place(asm, w.out, w.a.c0, w.a.c1, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 0, w.out, w.a, s);

    // c1 += ξ*(a2*b2)
    copy_fq2(asm, w.a, lhs.c2);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 128);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq2_mul_by_xi_in_place(asm, w.out, w.a.c0, w.a.c1, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 64, w.out, w.a, s);
}

/// Multiply `lhs` (in regs) by `rhs` (in regs) and write the Fq6 product to memory.
fn fq6_mul_schoolbook_regreg_to_mem(
    asm: &mut InstrAssembler,
    out_ptr: u8,
    out_off: i64,
    lhs: Fq6Regs,
    rhs: Fq6Regs,
    w: Fq2WorkTight,
    s: &FqScratch,
) {
    // c0 = a0*b0
    copy_fq2(asm, w.a, lhs.c0);
    copy_fq2(asm, w.b, rhs.c0);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    store_fq2(asm, out_ptr, out_off + 0, w.out);

    // c1 = a0*b1
    copy_fq2(asm, w.a, lhs.c0);
    copy_fq2(asm, w.b, rhs.c1);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    store_fq2(asm, out_ptr, out_off + 64, w.out);

    // c2 = a0*b2
    copy_fq2(asm, w.a, lhs.c0);
    copy_fq2(asm, w.b, rhs.c2);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    store_fq2(asm, out_ptr, out_off + 128, w.out);

    // c1 += a1*b0
    copy_fq2(asm, w.a, lhs.c1);
    copy_fq2(asm, w.b, rhs.c0);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 64, w.out, w.a, s);

    // c2 += a1*b1
    copy_fq2(asm, w.a, lhs.c1);
    copy_fq2(asm, w.b, rhs.c1);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 128, w.out, w.a, s);

    // c2 += a2*b0
    copy_fq2(asm, w.a, lhs.c2);
    copy_fq2(asm, w.b, rhs.c0);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 128, w.out, w.a, s);

    // c0 += ξ*(a1*b2)
    copy_fq2(asm, w.a, lhs.c1);
    copy_fq2(asm, w.b, rhs.c2);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq2_mul_by_xi_in_place(asm, w.out, w.a.c0, w.a.c1, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 0, w.out, w.a, s);

    // c0 += ξ*(a2*b1)
    copy_fq2(asm, w.a, lhs.c2);
    copy_fq2(asm, w.b, rhs.c1);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq2_mul_by_xi_in_place(asm, w.out, w.a.c0, w.a.c1, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 0, w.out, w.a, s);

    // c1 += ξ*(a2*b2)
    copy_fq2(asm, w.a, lhs.c2);
    copy_fq2(asm, w.b, rhs.c2);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq2_mul_by_xi_in_place(asm, w.out, w.a.c0, w.a.c1, s);
    fq6_add_fq2_into_mem(asm, out_ptr, out_off + 64, w.out, w.a, s);
}

/// Compute `Fq6(lhs) * Fq6(rhs)` and write the result into `out_ptr` as 3 Fq2s (24 limbs),
/// using `out_ptr` as scratch (requires at least 6 Fq2 slots = 384 bytes available).
///
/// Scratch slots (each 64 bytes / 8 limbs) at `out_ptr + out_off`:
/// - slot0: ad
/// - slot1: be
/// - slot2: cf
/// - slot3: x
/// - slot4: y
/// - slot5: z
fn fq6_mul_karatsuba_mem(
    asm: &mut InstrAssembler,
    out_ptr: u8,
    out_off: i64,
    lhs_ptr: u8,
    lhs_off: i64,
    rhs_ptr: u8,
    rhs_off: i64,
    w: Fq2Work,
    s: &FqScratch,
) {
    let slot = |i: i64| out_off + i * 64;

    // --- ad ---
    load_fq2(asm, w.a, lhs_ptr, lhs_off + 0);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 0);
    fq2_mul_karatsuba(asm, w.out, w.a, w.b, w.t0, w.t1, w.t2, s);
    store_fq2(asm, out_ptr, slot(0), w.out);

    // --- be ---
    load_fq2(asm, w.a, lhs_ptr, lhs_off + 64);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 64);
    fq2_mul_karatsuba(asm, w.out, w.a, w.b, w.t0, w.t1, w.t2, s);
    store_fq2(asm, out_ptr, slot(1), w.out);

    // --- cf ---
    load_fq2(asm, w.a, lhs_ptr, lhs_off + 128);
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 128);
    fq2_mul_karatsuba(asm, w.out, w.a, w.b, w.t0, w.t1, w.t2, s);
    store_fq2(asm, out_ptr, slot(2), w.out);

    // --- x = (e+f)*(b+c) - be - cf ---
    load_fq2(asm, w.a, lhs_ptr, lhs_off + 64); // e
    load_fq2(asm, w.b, lhs_ptr, lhs_off + 128); // f
    fq2_add_mod(asm, w.a, w.a, w.b, s); // a = e+f
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 64); // b
    load_fq2(asm, w.out, rhs_ptr, rhs_off + 128); // c
    fq2_add_mod(asm, w.b, w.b, w.out, s); // b = b+c
    fq2_mul_karatsuba(asm, w.out, w.a, w.b, w.t0, w.t1, w.t2, s); // out = (e+f)*(b+c)
    load_fq2(asm, w.a, out_ptr, slot(1)); // be
    fq2_sub_mod(asm, w.out, w.out, w.a, s);
    load_fq2(asm, w.a, out_ptr, slot(2)); // cf
    fq2_sub_mod(asm, w.out, w.out, w.a, s);
    store_fq2(asm, out_ptr, slot(3), w.out); // x

    // --- y = (d+e)*(a+b) - ad - be ---
    load_fq2(asm, w.a, lhs_ptr, lhs_off + 0); // d
    load_fq2(asm, w.b, lhs_ptr, lhs_off + 64); // e
    fq2_add_mod(asm, w.a, w.a, w.b, s); // a = d+e
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 0); // a
    load_fq2(asm, w.out, rhs_ptr, rhs_off + 64); // b
    fq2_add_mod(asm, w.b, w.b, w.out, s); // b = a+b
    fq2_mul_karatsuba(asm, w.out, w.a, w.b, w.t0, w.t1, w.t2, s);
    load_fq2(asm, w.a, out_ptr, slot(0)); // ad
    fq2_sub_mod(asm, w.out, w.out, w.a, s);
    load_fq2(asm, w.a, out_ptr, slot(1)); // be
    fq2_sub_mod(asm, w.out, w.out, w.a, s);
    store_fq2(asm, out_ptr, slot(4), w.out); // y

    // --- z = (d+f)*(a+c) - ad + be - cf ---
    load_fq2(asm, w.a, lhs_ptr, lhs_off + 0); // d
    load_fq2(asm, w.b, lhs_ptr, lhs_off + 128); // f
    fq2_add_mod(asm, w.a, w.a, w.b, s); // a = d+f
    load_fq2(asm, w.b, rhs_ptr, rhs_off + 0); // a
    load_fq2(asm, w.out, rhs_ptr, rhs_off + 128); // c
    fq2_add_mod(asm, w.b, w.b, w.out, s); // b = a+c
    fq2_mul_karatsuba(asm, w.out, w.a, w.b, w.t0, w.t1, w.t2, s);
    load_fq2(asm, w.a, out_ptr, slot(0)); // ad
    fq2_sub_mod(asm, w.out, w.out, w.a, s);
    load_fq2(asm, w.a, out_ptr, slot(1)); // be
    fq2_add_mod(asm, w.out, w.out, w.a, s);
    load_fq2(asm, w.a, out_ptr, slot(2)); // cf
    fq2_sub_mod(asm, w.out, w.out, w.a, s);
    store_fq2(asm, out_ptr, slot(5), w.out); // z

    // --- c0 = ad + ξ*x ---
    load_fq2(asm, w.a, out_ptr, slot(3)); // x
    fq2_mul_by_xi_in_place(asm, w.a, w.t0, w.t1, s);
    load_fq2(asm, w.b, out_ptr, slot(0)); // ad
    fq2_add_mod(asm, w.a, w.a, w.b, s);
    store_fq2(asm, out_ptr, slot(0), w.a); // c0

    // --- c1 = y + ξ*cf ---
    load_fq2(asm, w.a, out_ptr, slot(2)); // cf
    fq2_mul_by_xi_in_place(asm, w.a, w.t0, w.t1, s);
    load_fq2(asm, w.b, out_ptr, slot(4)); // y
    fq2_add_mod(asm, w.a, w.a, w.b, s);
    store_fq2(asm, out_ptr, slot(1), w.a); // c1 (slot1)

    // --- c2 = z ---
    load_fq2(asm, w.a, out_ptr, slot(5));
    store_fq2(asm, out_ptr, slot(2), w.a); // c2 (slot2)
}

/// Compute `out = a * b (mod p)` for BN254 Fq limbs in Montgomery form.
///
/// This matches arkworks Montgomery multiplication for `Fp256<MontBackend<FqConfig,4>>`.
fn fq_mul_mont(asm: &mut InstrAssembler, out: FqRegs, a: FqRegs, b: FqRegs, s: &FqScratch) {
    // Implement the same CIOS-style Montgomery multiplication used by arkworks' "no-carry" path:
    // montgomery_backend.rs `mul_assign` (for N=4).
    //
    // r = 0
    for limb in out.iter() {
        li(asm, *limb, 0);
    }

    // We keep r as out[0..4). This function assumes out does not alias a/b.
    // (Callers can copy if needed.)

    // For i in 0..4 (unrolled)
    for i in 0..4 {
        // carry1 = 0 (use t1 for carry1)
        li(asm, s.t1, 0);
        // r0 = mac(r0, a0, b[i], &carry1)
        fq_mac(asm, out[0], out[0], a[0], b[i], s.t1, s);

        // k = r0 * INV (low 64 bits)
        asm.emit_r::<MUL>(s.t0, out[0], s.inv); // t0 = k

        // carry2 = 0 (use t2 for carry2)
        li(asm, s.t2, 0);
        // mac_discard(r0, k, MOD0, &carry2)
        fq_mac_discard(asm, out[0], s.t0, s.mod_limb[0], s.t2, s);

        // j=1..3
        // r1 = mac_with_carry(r1, a1, b[i], &carry1)
        fq_mac_with_carry(asm, out[1], out[1], a[1], b[i], s.t1, s);
        // r0 = mac_with_carry(r1, k, MOD1, &carry2)
        fq_mac_with_carry(asm, out[0], out[1], s.t0, s.mod_limb[1], s.t2, s);

        fq_mac_with_carry(asm, out[2], out[2], a[2], b[i], s.t1, s);
        fq_mac_with_carry(asm, out[1], out[2], s.t0, s.mod_limb[2], s.t2, s);

        fq_mac_with_carry(asm, out[3], out[3], a[3], b[i], s.t1, s);
        fq_mac_with_carry(asm, out[2], out[3], s.t0, s.mod_limb[3], s.t2, s);

        // r3 = carry1 + carry2
        asm.emit_r::<ADD>(out[3], s.t1, s.t2);
    }

    // Final conditional subtract modulus (branchless):
    // out = out - p; if borrow==1, add p back.
    li(asm, s.t2, 0); // borrow (0/1)
    for i in 0..4 {
        // diff1 = out[i] - mod[i]
        sub_u64_set_borrow(asm, s.t3, out[i], s.mod_limb[i], s.c0, s.c2); // c0 = borrow1
                                                                          // out[i] = diff1 - borrow_in
        asm.emit_r::<SUB>(out[i], s.t3, s.t2);
        // borrow2 = diff1 < borrow_in
        asm.emit_r::<SLTU>(s.c1, s.t3, s.t2);
        // borrow_out = borrow1 | borrow2 (safe to add)
        asm.emit_r::<ADD>(s.t2, s.c0, s.c1);
    }

    // if borrow==1 add modulus back
    li(asm, s.c2, 0); // carry
    for i in 0..4 {
        asm.emit_r::<MUL>(s.t3, s.t2, s.mod_limb[i]); // t3 = borrow * mod[i]
                                                      // out[i] = out[i] + t3 + carry
        add_u64_set_carry(asm, out[i], out[i], s.t3, s.c0, s.t1);
        asm.emit_r::<ADD>(out[i], out[i], s.c2);
        asm.emit_r::<SLTU>(s.c1, out[i], s.c2);
        asm.emit_r::<ADD>(s.c2, s.c0, s.c1); // carry_out (0/1)
    }
}

// -----------------------------
// BN254 Fq12 helpers (for GT exponentiation)
// -----------------------------

#[derive(Clone, Copy)]
struct Fq12Regs {
    c0: Fq6Regs,
    c1: Fq6Regs,
}

#[inline(always)]
fn fq12_regs_from_flat(flat: &[u8; 48]) -> Fq12Regs {
    let fq2 = |base: usize| Fq2Regs {
        c0: [
            flat[base + 0],
            flat[base + 1],
            flat[base + 2],
            flat[base + 3],
        ],
        c1: [
            flat[base + 4],
            flat[base + 5],
            flat[base + 6],
            flat[base + 7],
        ],
    };
    Fq12Regs {
        c0: Fq6Regs {
            c0: fq2(0),
            c1: fq2(8),
            c2: fq2(16),
        },
        c1: Fq6Regs {
            c0: fq2(24),
            c1: fq2(32),
            c2: fq2(40),
        },
    }
}

/// In-place Fq12 squaring: `acc := acc^2`.
///
/// Uses `out_ptr[..384]` as scratch memory during computation.
fn fq12_square_in_place(
    asm: &mut InstrAssembler,
    acc: Fq12Regs,
    out_ptr: u8,
    w: Fq2WorkTight,
    s: &FqScratch,
) {
    // Layout in `out_ptr` scratch:
    // - [0..192): v2 = c0*c1
    // - [192..384): staging for c0_sq and c1_sq

    // v2 = c0*c1 -> out_ptr[0..192]
    fq6_mul_schoolbook_regreg_to_mem(asm, out_ptr, 0, acc.c0, acc.c1, w, s);

    // c0_sq = c0*c0 -> out_ptr[192..384]
    fq6_mul_schoolbook_regreg_to_mem(asm, out_ptr, 192, acc.c0, acc.c0, w, s);
    // load c0_sq into acc.c0 (overwrite)
    load_fq2(asm, w.a, out_ptr, 192 + 0);
    copy_fq2(asm, acc.c0.c0, w.a);
    load_fq2(asm, w.a, out_ptr, 192 + 64);
    copy_fq2(asm, acc.c0.c1, w.a);
    load_fq2(asm, w.a, out_ptr, 192 + 128);
    copy_fq2(asm, acc.c0.c2, w.a);

    // c1_sq = c1*c1 -> out_ptr[192..384] (overwrite staging)
    fq6_mul_schoolbook_regreg_to_mem(asm, out_ptr, 192, acc.c1, acc.c1, w, s);
    // load c1_sq into acc.c1 (overwrite)
    load_fq2(asm, w.a, out_ptr, 192 + 0);
    copy_fq2(asm, acc.c1.c0, w.a);
    load_fq2(asm, w.a, out_ptr, 192 + 64);
    copy_fq2(asm, acc.c1.c1, w.a);
    load_fq2(asm, w.a, out_ptr, 192 + 128);
    copy_fq2(asm, acc.c1.c2, w.a);

    // acc.c0 = c0_sq + v*c1_sq, where v*(x0,x1,x2) = (x2*xi, x0, x1)
    copy_fq2(asm, w.a, acc.c1.c2);
    fq2_mul_by_xi_in_place(asm, w.a, w.b.c0, w.b.c1, s);
    fq2_add_mod(asm, acc.c0.c0, acc.c0.c0, w.a, s);
    fq2_add_mod(asm, acc.c0.c1, acc.c0.c1, acc.c1.c0, s);
    fq2_add_mod(asm, acc.c0.c2, acc.c0.c2, acc.c1.c1, s);

    // acc.c1 = 2*v2 (load v2 from out_ptr[0..192])
    load_fq2(asm, w.a, out_ptr, 0 + 0);
    fq2_add_mod(asm, w.a, w.a, w.a, s);
    copy_fq2(asm, acc.c1.c0, w.a);
    load_fq2(asm, w.a, out_ptr, 0 + 64);
    fq2_add_mod(asm, w.a, w.a, w.a, s);
    copy_fq2(asm, acc.c1.c1, w.a);
    load_fq2(asm, w.a, out_ptr, 0 + 128);
    fq2_add_mod(asm, w.a, w.a, w.a, s);
    copy_fq2(asm, acc.c1.c2, w.a);
}

/// In-place *cyclotomic* Fq12 squaring: `acc := acc^2`, specialized for BN254 GT elements.
///
/// This implements the same algorithm as arkworks'
/// `Fp12::<P>::cyclotomic_square_in_place()` for the `p^2 mod 6 == 1` case
/// (Granger-Scott "Faster Squaring in the Cyclotomic Subgroup of Sixth Degree Extensions").
///
/// # Safety / correctness contract
/// This is only correct when `acc` is in the cyclotomic subgroup of Fq12 (which includes BN254 GT).
///
/// # Scratch
/// Uses `out_ptr[..384]` as scratch memory to store intermediate Fq2 values.
fn fq12_cyclotomic_square_in_place(
    asm: &mut InstrAssembler,
    acc: Fq12Regs,
    out_ptr: u8,
    w: Fq2WorkTight,
    s: &FqScratch,
) {
    // Coefficient mapping matches arkworks' `fp12_2over3over2.rs`:
    // r0 = c0.c0, r4 = c0.c1, r3 = c0.c2, r2 = c1.c0, r1 = c1.c1, r5 = c1.c2.
    let r0 = acc.c0.c0;
    let r4 = acc.c0.c1;
    let r3 = acc.c0.c2;
    let r2 = acc.c1.c0;
    let r1 = acc.c1.c1;
    let r5 = acc.c1.c2;

    // Scratch layout in out_ptr (6 x Fq2 = 384 bytes):
    // slot0: t0, slot1: t1, slot2: t2, slot3: t3, slot4: t4, slot5: t5.
    let slot = |i: i64| i * 64;

    // Helper: compute (t_even, t_odd) for a pair (a,b) and store into (even_slot, odd_slot).
    //
    // Given (a, b) in Fq2, compute:
    //   tmp = a*b
    //   t_even = (a + b) * (xi*b + a) - tmp - xi*tmp
    //   t_odd  = 2*tmp
    // where xi is the Fq6 nonresidue (BN254: 9 + u).
    let mut pair = |a: Fq2Regs, b: Fq2Regs, even_slot: i64, odd_slot: i64| {
        // tmp = a*b
        copy_fq2(asm, w.a, a);
        copy_fq2(asm, w.b, b);
        fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);

        // Store tmp to even_slot (will be overwritten by t_even later).
        store_fq2(asm, out_ptr, even_slot, w.out);

        // t_odd = 2*tmp
        copy_fq2(asm, w.a, w.out);
        fq2_double_in_place(asm, w.a, s);
        store_fq2(asm, out_ptr, odd_slot, w.a);

        // y = xi*b + a  (compute in w.b; use w.a as temps for mul_by_xi)
        copy_fq2(asm, w.b, b);
        fq2_mul_by_xi_in_place(asm, w.b, w.a.c0, w.a.c1, s);
        fq2_add_mod(asm, w.b, w.b, a, s);

        // x = a + b  (compute in w.a)
        copy_fq2(asm, w.a, a);
        fq2_add_mod(asm, w.a, w.a, b, s);

        // prod = x*y
        fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);

        // prod -= tmp
        load_fq2(asm, w.a, out_ptr, even_slot);
        fq2_sub_mod(asm, w.out, w.out, w.a, s);

        // prod -= xi*tmp
        copy_fq2(asm, w.b, w.a);
        fq2_mul_by_xi_in_place(asm, w.b, w.a.c0, w.a.c1, s); // clobbers w.a; ok
        fq2_sub_mod(asm, w.out, w.out, w.b, s);

        // Store t_even.
        store_fq2(asm, out_ptr, even_slot, w.out);
    };

    // Compute t0..t5.
    // (r0, r1) -> (t0, t1)
    pair(r0, r1, slot(0), slot(1));
    // (r2, r3) -> (t2, t3)
    pair(r2, r3, slot(2), slot(3));
    // (r4, r5) -> (t4, t5)
    pair(r4, r5, slot(4), slot(5));

    // Update coefficients in-place.
    let z0 = acc.c0.c0;
    let z4 = acc.c0.c1;
    let z3 = acc.c0.c2;
    let z2 = acc.c1.c0;
    let z1 = acc.c1.c1;
    let z5 = acc.c1.c2;

    // z0 = 3*t0 - 2*z0
    load_fq2(asm, w.a, out_ptr, slot(0));
    fq2_sub_mod(asm, z0, w.a, z0, s);
    fq2_double_in_place(asm, z0, s);
    fq2_add_mod(asm, z0, z0, w.a, s);

    // z1 = 3*t1 + 2*z1
    load_fq2(asm, w.a, out_ptr, slot(1));
    fq2_add_mod(asm, z1, w.a, z1, s);
    fq2_double_in_place(asm, z1, s);
    fq2_add_mod(asm, z1, z1, w.a, s);

    // z2 = 3*(xi*t5) + 2*z2
    load_fq2(asm, w.b, out_ptr, slot(5));
    fq2_mul_by_xi_in_place(asm, w.b, w.a.c0, w.a.c1, s);
    fq2_add_mod(asm, z2, z2, w.b, s);
    fq2_double_in_place(asm, z2, s);
    fq2_add_mod(asm, z2, z2, w.b, s);

    // z3 = 3*t4 - 2*z3
    load_fq2(asm, w.a, out_ptr, slot(4));
    fq2_sub_mod(asm, z3, w.a, z3, s);
    fq2_double_in_place(asm, z3, s);
    fq2_add_mod(asm, z3, z3, w.a, s);

    // z4 = 3*t2 - 2*z4
    load_fq2(asm, w.a, out_ptr, slot(2));
    fq2_sub_mod(asm, z4, w.a, z4, s);
    fq2_double_in_place(asm, z4, s);
    fq2_add_mod(asm, z4, z4, w.a, s);

    // z5 = 3*t3 + 2*z5
    load_fq2(asm, w.a, out_ptr, slot(3));
    fq2_add_mod(asm, z5, z5, w.a, s);
    fq2_double_in_place(asm, z5, s);
    fq2_add_mod(asm, z5, z5, w.a, s);
}

/// Fq12 multiplication: `out_ptr[..384] := lhs * rhs`, where `lhs` is in regs and `rhs` is in memory.
///
/// Uses the quadratic-extension identity at the Fq12 layer, and the schoolbook Fq6 mul routines.
fn fq12_mul_regmem_to_mem(
    asm: &mut InstrAssembler,
    out_ptr: u8,
    lhs: Fq12Regs,
    rhs_ptr: u8,
    rhs_off: i64,
    w: Fq2WorkTight,
    s: &FqScratch,
) {
    let c0_off = 0i64;
    let c1_off = 192i64;

    // Fq12 Karatsuba (quadratic extension over Fq6):
    //   t0 = a0*b0
    //   t1 = a1*b1
    //   t2 = (a0+a1)*(b0+b1)
    //   c0 = t0 + v*t1
    //   c1 = t2 - t0 - t1
    //
    // We avoid materializing all of t1 in memory by directly accumulating its contributions:
    // - into c0 via the v-mapping v*(x0,x1,x2) = (x2*xi, x0, x1)
    // - into c1 via subtraction
    //
    // Scratch strategy:
    // - use out.c0 temporarily to store (b0+b1) while computing t2
    // - store t2 in out.c1
    // - overwrite out.c0 with t0
    // - update out.c1 in-place to (t2 - t0 - t1) and out.c0 to (t0 + v*t1)

    let rhs_b0_off = rhs_off + 0;
    let rhs_b1_off = rhs_off + 192;

    // out.c0 := (b0 + b1)  (temporary scratch)
    for fq2_i in 0..3i64 {
        let off = fq2_i * 64;
        load_fq2(asm, w.a, rhs_ptr, rhs_b0_off + off);
        load_fq2(asm, w.b, rhs_ptr, rhs_b1_off + off);
        fq2_add_mod(asm, w.out, w.a, w.b, s);
        store_fq2(asm, out_ptr, c0_off + off, w.out);
    }

    // lhs.c0 := a0 + a1 (temporary, restored immediately after computing t2)
    fq6_add_mod(asm, lhs.c0, lhs.c0, lhs.c1, s);

    // t2 := (a0+a1)*(b0+b1) -> out.c1
    fq6_mul_schoolbook_regmem_to_mem(asm, out_ptr, c1_off, lhs.c0, out_ptr, c0_off, w, s);

    // Restore lhs.c0 := (a0+a1) - a1 = a0
    fq6_sub_mod(asm, lhs.c0, lhs.c0, lhs.c1, s);

    // t0 := a0*b0 -> out.c0 (overwrites b0+b1 scratch)
    fq6_mul_schoolbook_regmem_to_mem(asm, out_ptr, c0_off, lhs.c0, rhs_ptr, rhs_b0_off, w, s);

    // out.c1 := out.c1 - out.c0  (t2 - t0)
    for fq2_i in 0..3i64 {
        let off = fq2_i * 64;
        load_fq2(asm, w.a, out_ptr, c1_off + off);
        load_fq2(asm, w.b, out_ptr, c0_off + off);
        fq2_sub_mod(asm, w.out, w.a, w.b, s);
        store_fq2(asm, out_ptr, c1_off + off, w.out);
    }

    // Incorporate t1 := a1*b1 into:
    // - c1 via subtraction: out.c1 -= t1
    // - c0 via v-mapping:   out.c0 += v*t1
    //
    // We do this term-by-term (schoolbook) without ever storing full t1.
    let b1_ptr = rhs_ptr;
    let b1_off = rhs_b1_off;

    // --- t1.c0 contributions (update c1.c0 and c0.c1) ---
    // c0_term = a0*b0
    copy_fq2(asm, w.a, lhs.c1.c0);
    load_fq2(asm, w.b, b1_ptr, b1_off + 0);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_sub_fq2_into_mem(asm, out_ptr, c1_off + 0, w.out, w.a, s);
    fq6_add_fq2_into_mem(asm, out_ptr, c0_off + 64, w.out, w.a, s);

    // c0_term += xi*(a1*b2)
    copy_fq2(asm, w.a, lhs.c1.c1);
    load_fq2(asm, w.b, b1_ptr, b1_off + 128);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq2_mul_by_xi_in_place(asm, w.out, w.a.c0, w.a.c1, s);
    fq6_sub_fq2_into_mem(asm, out_ptr, c1_off + 0, w.out, w.a, s);
    fq6_add_fq2_into_mem(asm, out_ptr, c0_off + 64, w.out, w.a, s);

    // c0_term += xi*(a2*b1)
    copy_fq2(asm, w.a, lhs.c1.c2);
    load_fq2(asm, w.b, b1_ptr, b1_off + 64);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq2_mul_by_xi_in_place(asm, w.out, w.a.c0, w.a.c1, s);
    fq6_sub_fq2_into_mem(asm, out_ptr, c1_off + 0, w.out, w.a, s);
    fq6_add_fq2_into_mem(asm, out_ptr, c0_off + 64, w.out, w.a, s);

    // --- t1.c1 contributions (update c1.c1 and c0.c2) ---
    // c1_term = a0*b1
    copy_fq2(asm, w.a, lhs.c1.c0);
    load_fq2(asm, w.b, b1_ptr, b1_off + 64);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_sub_fq2_into_mem(asm, out_ptr, c1_off + 64, w.out, w.a, s);
    fq6_add_fq2_into_mem(asm, out_ptr, c0_off + 128, w.out, w.a, s);

    // c1_term += a1*b0
    copy_fq2(asm, w.a, lhs.c1.c1);
    load_fq2(asm, w.b, b1_ptr, b1_off + 0);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_sub_fq2_into_mem(asm, out_ptr, c1_off + 64, w.out, w.a, s);
    fq6_add_fq2_into_mem(asm, out_ptr, c0_off + 128, w.out, w.a, s);

    // c1_term += xi*(a2*b2)
    copy_fq2(asm, w.a, lhs.c1.c2);
    load_fq2(asm, w.b, b1_ptr, b1_off + 128);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq2_mul_by_xi_in_place(asm, w.out, w.a.c0, w.a.c1, s);
    fq6_sub_fq2_into_mem(asm, out_ptr, c1_off + 64, w.out, w.a, s);
    fq6_add_fq2_into_mem(asm, out_ptr, c0_off + 128, w.out, w.a, s);

    // --- t1.c2 contributions (update c1.c2 and c0.c0 via xi) ---
    // c2_term = a0*b2
    copy_fq2(asm, w.a, lhs.c1.c0);
    load_fq2(asm, w.b, b1_ptr, b1_off + 128);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_sub_fq2_into_mem(asm, out_ptr, c1_off + 128, w.out, w.a, s);
    fq2_mul_by_xi_in_place(asm, w.out, w.a.c0, w.a.c1, s);
    fq6_add_fq2_into_mem(asm, out_ptr, c0_off + 0, w.out, w.a, s);

    // c2_term += a1*b1
    copy_fq2(asm, w.a, lhs.c1.c1);
    load_fq2(asm, w.b, b1_ptr, b1_off + 64);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_sub_fq2_into_mem(asm, out_ptr, c1_off + 128, w.out, w.a, s);
    fq2_mul_by_xi_in_place(asm, w.out, w.a.c0, w.a.c1, s);
    fq6_add_fq2_into_mem(asm, out_ptr, c0_off + 0, w.out, w.a, s);

    // c2_term += a2*b0
    copy_fq2(asm, w.a, lhs.c1.c2);
    load_fq2(asm, w.b, b1_ptr, b1_off + 0);
    fq2_mul_karatsuba_clobber(asm, w.out, w.a, w.b, s);
    fq6_sub_fq2_into_mem(asm, out_ptr, c1_off + 128, w.out, w.a, s);
    fq2_mul_by_xi_in_place(asm, w.out, w.a.c0, w.a.c1, s);
    fq6_add_fq2_into_mem(asm, out_ptr, c0_off + 0, w.out, w.a, s);
}

// -----------------------------
// BN254_GT_EXP inline entrypoint (still placeholder for now)
// -----------------------------

/// Sequence builder for `BN254_GT_EXP`.
///
/// Correctness-first implementation:
/// - Fixed 256-bit MSB→LSB schedule (no control-flow inside the inline expansion).
/// - Branchless conditional multiply using a delta update:
///   `acc_next = acc_sq + bit * (acc_sq*base - acc_sq)`.
/// - Uses `out_ptr[..384]` as scratch during computation.
/// - Temporarily rejects the aliasing case `out_ptr == base_ptr` (see assertion below).
pub fn bn254_gt_exp_sequence_builder(
    mut asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    // Allocate base-field scratch (14 regs) and initialize constants.
    let fq_scratch_vr: [VirtualRegisterGuard; 14] =
        array::from_fn(|_| asm.allocator.allocate_for_inline());
    let fq_scratch = FqScratch::new(&fq_scratch_vr);
    fq_scratch.init_constants(&mut asm);

    // Allocate tight Fq2 work regs (24 regs) for Fq6/Fq12 ops.
    let work_a_vr: [VirtualRegisterGuard; 8] =
        array::from_fn(|_| asm.allocator.allocate_for_inline());
    let work_b_vr: [VirtualRegisterGuard; 8] =
        array::from_fn(|_| asm.allocator.allocate_for_inline());
    let work_out_vr: [VirtualRegisterGuard; 8] =
        array::from_fn(|_| asm.allocator.allocate_for_inline());
    let work = Fq2WorkTight {
        a: Fq2Regs {
            c0: [*work_a_vr[0], *work_a_vr[1], *work_a_vr[2], *work_a_vr[3]],
            c1: [*work_a_vr[4], *work_a_vr[5], *work_a_vr[6], *work_a_vr[7]],
        },
        b: Fq2Regs {
            c0: [*work_b_vr[0], *work_b_vr[1], *work_b_vr[2], *work_b_vr[3]],
            c1: [*work_b_vr[4], *work_b_vr[5], *work_b_vr[6], *work_b_vr[7]],
        },
        out: Fq2Regs {
            c0: [
                *work_out_vr[0],
                *work_out_vr[1],
                *work_out_vr[2],
                *work_out_vr[3],
            ],
            c1: [
                *work_out_vr[4],
                *work_out_vr[5],
                *work_out_vr[6],
                *work_out_vr[7],
            ],
        },
    };

    // Accumulator in registers (48 limbs in ABI flatten order).
    let acc_vr: [VirtualRegisterGuard; 48] =
        array::from_fn(|_| asm.allocator.allocate_for_inline());
    let acc_flat: [u8; 48] = array::from_fn(|i| *acc_vr[i]);
    let acc = fq12_regs_from_flat(&acc_flat);

    // Dedicated exponent-limb register (do not let field ops clobber it).
    let exp_limb_vr: VirtualRegisterGuard = asm.allocator.allocate_for_inline();
    let exp_limb = *exp_limb_vr;

    let exp_ptr = operands.rs1;
    let base_ptr = operands.rs2;
    let out_ptr = operands.rs3;

    // Reject aliasing: base_ptr must not equal out_ptr (we overwrite out_ptr as scratch/output).
    // Assert `base_ptr != out_ptr` via: (base < out) + (out < base) == 1.
    asm.emit_r::<SLTU>(fq_scratch.c0, base_ptr, out_ptr);
    asm.emit_r::<SLTU>(fq_scratch.c1, out_ptr, base_ptr);
    asm.emit_r::<ADD>(fq_scratch.c2, fq_scratch.c0, fq_scratch.c1);
    li(&mut asm, fq_scratch.t3, 1);
    asm.emit_b::<VirtualAssertEQ>(fq_scratch.c2, fq_scratch.t3, 0);

    // Initialize acc = 1 ∈ Fq12 (Montgomery form).
    for i in 0..48 {
        li(&mut asm, acc_flat[i], 0);
    }
    li(&mut asm, acc_flat[0], BN254_FQ_ONE[0]);
    li(&mut asm, acc_flat[1], BN254_FQ_ONE[1]);
    li(&mut asm, acc_flat[2], BN254_FQ_ONE[2]);
    li(&mut asm, acc_flat[3], BN254_FQ_ONE[3]);

    // Fixed 256-bit schedule: iterate exponent bits from MS limb→LS limb, MSB→LSB.
    for limb_idx in (0..4usize).rev() {
        asm.emit_ld::<LD>(exp_limb, exp_ptr, (limb_idx as i64) * 8);
        for bit_idx in (0..64usize).rev() {
            // acc := acc^2
            fq12_square_in_place(&mut asm, acc, out_ptr, work, &fq_scratch);

            // out_ptr := acc * base
            fq12_mul_regmem_to_mem(&mut asm, out_ptr, acc, base_ptr, 0, work, &fq_scratch);

            // out_ptr := out_ptr - acc   (delta for branchless conditional multiply)
            for fq_i in 0..12usize {
                let mem_off = (fq_i as i64) * 32;
                let acc_fq: FqRegs = [
                    acc_flat[fq_i * 4 + 0],
                    acc_flat[fq_i * 4 + 1],
                    acc_flat[fq_i * 4 + 2],
                    acc_flat[fq_i * 4 + 3],
                ];
                load_fq(&mut asm, work.a.c0, out_ptr, mem_off);
                copy_fq(&mut asm, work.b.c0, acc_fq);
                fq_sub_mod(&mut asm, work.out.c0, work.a.c0, work.b.c0, &fq_scratch);
                store_fq(&mut asm, out_ptr, mem_off, work.out.c0);
            }

            // bit := ((exp_limb >> bit_idx) & 1)   (computed branchlessly without AND)
            if bit_idx == 63 {
                asm.emit_i::<SRLI>(fq_scratch.t0, exp_limb, 63);
            } else {
                asm.emit_i::<SRLI>(fq_scratch.t0, exp_limb, bit_idx as u64);
                asm.emit_i::<SRLI>(fq_scratch.t1, exp_limb, (bit_idx + 1) as u64);
                asm.emit_r::<ADD>(fq_scratch.t1, fq_scratch.t1, fq_scratch.t1);
                asm.emit_r::<SUB>(fq_scratch.t0, fq_scratch.t0, fq_scratch.t1);
            }

            // acc := acc + bit * delta
            for fq_i in 0..12usize {
                let mem_off = (fq_i as i64) * 32;
                let acc_fq: FqRegs = [
                    acc_flat[fq_i * 4 + 0],
                    acc_flat[fq_i * 4 + 1],
                    acc_flat[fq_i * 4 + 2],
                    acc_flat[fq_i * 4 + 3],
                ];
                load_fq(&mut asm, work.a.c0, out_ptr, mem_off);
                for i in 0..4 {
                    asm.emit_r::<MUL>(work.a.c0[i], work.a.c0[i], fq_scratch.t0);
                }
                fq_add_mod(&mut asm, acc_fq, acc_fq, work.a.c0, &fq_scratch);
            }
        }
    }

    // Store final accumulator to out_ptr.
    for i in 0..48 {
        asm.emit_s::<SD>(out_ptr, acc_flat[i], (i as i64) * 8);
    }

    // Cleanup: drop all guards before finalizing.
    drop(exp_limb_vr);
    drop(acc_vr);
    drop(work_a_vr);
    drop(work_b_vr);
    drop(work_out_vr);
    drop(fq_scratch_vr);

    asm.finalize_inline()
}

/// Sequence builder for `BN254_GT_MUL`.
///
/// Computes `out := lhs * rhs` where:
/// - `rs1` = `lhs_ptr` (48 x u64 limbs, Montgomery form)
/// - `rs2` = `rhs_ptr` (48 x u64 limbs, Montgomery form)
/// - `rs3` = `out_ptr` (48 x u64 limbs, Montgomery form)
///
/// Aliasing: we currently reject `out_ptr == rhs_ptr` because the implementation streams `rhs`
/// from memory while writing `out` into the same region.
pub fn bn254_gt_mul_sequence_builder(
    mut asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    // Allocate base-field scratch (14 regs) and initialize constants.
    let fq_scratch_vr: [VirtualRegisterGuard; 14] =
        array::from_fn(|_| asm.allocator.allocate_for_inline());
    let fq_scratch = FqScratch::new(&fq_scratch_vr);
    fq_scratch.init_constants(&mut asm);

    // Allocate tight Fq2 work regs (24 regs) for Fq6/Fq12 ops.
    let work_a_vr: [VirtualRegisterGuard; 8] =
        array::from_fn(|_| asm.allocator.allocate_for_inline());
    let work_b_vr: [VirtualRegisterGuard; 8] =
        array::from_fn(|_| asm.allocator.allocate_for_inline());
    let work_out_vr: [VirtualRegisterGuard; 8] =
        array::from_fn(|_| asm.allocator.allocate_for_inline());
    let work = Fq2WorkTight {
        a: Fq2Regs {
            c0: [*work_a_vr[0], *work_a_vr[1], *work_a_vr[2], *work_a_vr[3]],
            c1: [*work_a_vr[4], *work_a_vr[5], *work_a_vr[6], *work_a_vr[7]],
        },
        b: Fq2Regs {
            c0: [*work_b_vr[0], *work_b_vr[1], *work_b_vr[2], *work_b_vr[3]],
            c1: [*work_b_vr[4], *work_b_vr[5], *work_b_vr[6], *work_b_vr[7]],
        },
        out: Fq2Regs {
            c0: [
                *work_out_vr[0],
                *work_out_vr[1],
                *work_out_vr[2],
                *work_out_vr[3],
            ],
            c1: [
                *work_out_vr[4],
                *work_out_vr[5],
                *work_out_vr[6],
                *work_out_vr[7],
            ],
        },
    };

    // LHS in registers (48 limbs in ABI flatten order).
    let lhs_vr: [VirtualRegisterGuard; 48] =
        array::from_fn(|_| asm.allocator.allocate_for_inline());
    let lhs_flat: [u8; 48] = array::from_fn(|i| *lhs_vr[i]);
    let lhs = fq12_regs_from_flat(&lhs_flat);

    let lhs_ptr = operands.rs1;
    let rhs_ptr = operands.rs2;
    let out_ptr = operands.rs3;

    // Reject aliasing: rhs_ptr must not equal out_ptr (we stream rhs from memory while writing out).
    // Assert `rhs_ptr != out_ptr` via: (rhs < out) + (out < rhs) == 1.
    asm.emit_r::<SLTU>(fq_scratch.c0, rhs_ptr, out_ptr);
    asm.emit_r::<SLTU>(fq_scratch.c1, out_ptr, rhs_ptr);
    asm.emit_r::<ADD>(fq_scratch.c2, fq_scratch.c0, fq_scratch.c1);
    li(&mut asm, fq_scratch.t3, 1);
    asm.emit_b::<VirtualAssertEQ>(fq_scratch.c2, fq_scratch.t3, 0);

    // Load lhs into registers.
    for i in 0..48 {
        asm.emit_ld::<LD>(lhs_flat[i], lhs_ptr, (i as i64) * 8);
    }

    // out := lhs * rhs
    fq12_mul_regmem_to_mem(&mut asm, out_ptr, lhs, rhs_ptr, 0, work, &fq_scratch);

    // Cleanup: drop all guards before finalizing.
    drop(lhs_vr);
    drop(work_a_vr);
    drop(work_b_vr);
    drop(work_out_vr);
    drop(fq_scratch_vr);

    asm.finalize_inline()
}

/// Sequence builder for `BN254_GT_SQR`.
///
/// Computes `out := in^2` where:
/// - `rs1` = `in_ptr`  (48 x u64 limbs, Montgomery form)
/// - `rs3` = `out_ptr` (48 x u64 limbs, Montgomery form)
///
/// The instruction encoding duplicates `in_ptr` into `rs2`, but the builder ignores `rs2`.
pub fn bn254_gt_sqr_sequence_builder(
    mut asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    // Allocate base-field scratch (14 regs) and initialize constants.
    let fq_scratch_vr: [VirtualRegisterGuard; 14] =
        array::from_fn(|_| asm.allocator.allocate_for_inline());
    let fq_scratch = FqScratch::new(&fq_scratch_vr);
    fq_scratch.init_constants(&mut asm);

    // Allocate tight Fq2 work regs (24 regs) for Fq6/Fq12 ops.
    let work_a_vr: [VirtualRegisterGuard; 8] =
        array::from_fn(|_| asm.allocator.allocate_for_inline());
    let work_b_vr: [VirtualRegisterGuard; 8] =
        array::from_fn(|_| asm.allocator.allocate_for_inline());
    let work_out_vr: [VirtualRegisterGuard; 8] =
        array::from_fn(|_| asm.allocator.allocate_for_inline());
    let work = Fq2WorkTight {
        a: Fq2Regs {
            c0: [*work_a_vr[0], *work_a_vr[1], *work_a_vr[2], *work_a_vr[3]],
            c1: [*work_a_vr[4], *work_a_vr[5], *work_a_vr[6], *work_a_vr[7]],
        },
        b: Fq2Regs {
            c0: [*work_b_vr[0], *work_b_vr[1], *work_b_vr[2], *work_b_vr[3]],
            c1: [*work_b_vr[4], *work_b_vr[5], *work_b_vr[6], *work_b_vr[7]],
        },
        out: Fq2Regs {
            c0: [
                *work_out_vr[0],
                *work_out_vr[1],
                *work_out_vr[2],
                *work_out_vr[3],
            ],
            c1: [
                *work_out_vr[4],
                *work_out_vr[5],
                *work_out_vr[6],
                *work_out_vr[7],
            ],
        },
    };

    // Input/accumulator in registers (48 limbs in ABI flatten order).
    let acc_vr: [VirtualRegisterGuard; 48] =
        array::from_fn(|_| asm.allocator.allocate_for_inline());
    let acc_flat: [u8; 48] = array::from_fn(|i| *acc_vr[i]);
    let acc = fq12_regs_from_flat(&acc_flat);

    let in_ptr = operands.rs1;
    let out_ptr = operands.rs3;

    // Load input into registers.
    for i in 0..48 {
        asm.emit_ld::<LD>(acc_flat[i], in_ptr, (i as i64) * 8);
    }

    // acc := acc^2 (cyclotomic, GT-specialized; uses out_ptr as scratch during computation)
    fq12_cyclotomic_square_in_place(&mut asm, acc, out_ptr, work, &fq_scratch);

    // Store result to out_ptr.
    for i in 0..48 {
        asm.emit_s::<SD>(out_ptr, acc_flat[i], (i as i64) * 8);
    }

    // Cleanup: drop all guards before finalizing.
    drop(acc_vr);
    drop(work_a_vr);
    drop(work_b_vr);
    drop(work_out_vr);
    drop(fq_scratch_vr);

    asm.finalize_inline()
}

#[cfg(all(test, feature = "host"))]
mod tests {
    use super::*;

    use ark_bn254::{Bn254, Fq, Fq12, Fq2, Fq6, Fq6Config, G1Projective, G2Projective};
    use ark_ec::pairing::Pairing;
    use ark_ff::{AdditiveGroup, Field, Fp6Config, UniformRand};
    use ark_std::test_rng;
    use tracer::emulator::cpu::Xlen;
    use tracer::utils::inline_test_harness::{
        InlineMemoryLayout, InlineTestHarness, INLINE_RS1, INLINE_RS2, INLINE_RS3,
    };

    fn fq_to_limbs_mont(x: &Fq) -> [u64; 4] {
        // arkworks Fp elements store the Montgomery form in the inner BigInt.
        x.0 .0
    }

    fn fq2_to_limbs_mont(x: &Fq2) -> [u64; 8] {
        let c0 = fq_to_limbs_mont(&x.c0);
        let c1 = fq_to_limbs_mont(&x.c1);
        [c0[0], c0[1], c0[2], c0[3], c1[0], c1[1], c1[2], c1[3]]
    }

    fn fq6_to_limbs_mont(x: &Fq6) -> [u64; 24] {
        let c0 = fq2_to_limbs_mont(&x.c0);
        let c1 = fq2_to_limbs_mont(&x.c1);
        let c2 = fq2_to_limbs_mont(&x.c2);
        [
            c0[0], c0[1], c0[2], c0[3], c0[4], c0[5], c0[6], c0[7], c1[0], c1[1], c1[2], c1[3],
            c1[4], c1[5], c1[6], c1[7], c2[0], c2[1], c2[2], c2[3], c2[4], c2[5], c2[6], c2[7],
        ]
    }

    fn fq12_to_limbs_mont(x: &Fq12) -> [u64; 48] {
        let c0 = fq6_to_limbs_mont(&x.c0);
        let c1 = fq6_to_limbs_mont(&x.c1);
        let mut out = [0u64; 48];
        out[0..24].copy_from_slice(&c0);
        out[24..48].copy_from_slice(&c1);
        out
    }

    fn run_fq_mul_inline(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        // Layout: rs1=input (32 bytes), rs2=input2 (32 bytes), rs3=output (32 bytes)
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();
        harness.load_input64(&a);
        harness.load_input2_64(&b);

        // Build instruction sequence: load a,b; fq_mul_mont; store out.
        let mut asm =
            InstrAssembler::new_inline(0, false, harness.xlen(), &harness.cpu.vr_allocator);

        // Allocate regs: a(4), b(4), out(4), scratch(14).
        let a_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let b_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let out_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let scratch_vr: [VirtualRegisterGuard; 14] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let scratch = FqScratch::new(&scratch_vr);
        scratch.init_constants(&mut asm);

        // Load limbs from memory.
        for i in 0..4 {
            asm.emit_ld::<tracer::instruction::ld::LD>(*a_vr[i], INLINE_RS1, (i as i64) * 8);
            asm.emit_ld::<tracer::instruction::ld::LD>(*b_vr[i], INLINE_RS2, (i as i64) * 8);
        }

        // Compute out = a*b.
        fq_mul_mont(
            &mut asm,
            [*out_vr[0], *out_vr[1], *out_vr[2], *out_vr[3]],
            [*a_vr[0], *a_vr[1], *a_vr[2], *a_vr[3]],
            [*b_vr[0], *b_vr[1], *b_vr[2], *b_vr[3]],
            &scratch,
        );

        // Store result.
        for i in 0..4 {
            asm.emit_s::<tracer::instruction::sd::SD>(INLINE_RS3, *out_vr[i], (i as i64) * 8);
        }

        drop(a_vr);
        drop(b_vr);
        drop(out_vr);
        drop(scratch_vr);
        let sequence = asm.finalize_inline();

        harness.execute_sequence(&sequence);
        let out_vec = harness.read_output64(4);
        [out_vec[0], out_vec[1], out_vec[2], out_vec[3]]
    }

    #[test]
    fn fq_mul_matches_arkworks_montgomery_limbs() {
        let mut rng = test_rng();
        for _ in 0..50 {
            let a = Fq::rand(&mut rng);
            let b = Fq::rand(&mut rng);
            let expected = fq_to_limbs_mont(&(a * b));
            let got = run_fq_mul_inline(fq_to_limbs_mont(&a), fq_to_limbs_mont(&b));
            assert_eq!(got, expected, "Fq mul mismatch");
        }
    }

    fn run_fq_add_inline(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();
        harness.load_input64(&a);
        harness.load_input2_64(&b);

        let mut asm =
            InstrAssembler::new_inline(0, false, harness.xlen(), &harness.cpu.vr_allocator);
        let a_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let b_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let out_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let scratch_vr: [VirtualRegisterGuard; 14] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let scratch = FqScratch::new(&scratch_vr);
        scratch.init_constants(&mut asm);

        for i in 0..4 {
            asm.emit_ld::<tracer::instruction::ld::LD>(*a_vr[i], INLINE_RS1, (i as i64) * 8);
            asm.emit_ld::<tracer::instruction::ld::LD>(*b_vr[i], INLINE_RS2, (i as i64) * 8);
        }

        fq_add_mod(
            &mut asm,
            [*out_vr[0], *out_vr[1], *out_vr[2], *out_vr[3]],
            [*a_vr[0], *a_vr[1], *a_vr[2], *a_vr[3]],
            [*b_vr[0], *b_vr[1], *b_vr[2], *b_vr[3]],
            &scratch,
        );

        for i in 0..4 {
            asm.emit_s::<tracer::instruction::sd::SD>(INLINE_RS3, *out_vr[i], (i as i64) * 8);
        }

        drop(a_vr);
        drop(b_vr);
        drop(out_vr);
        drop(scratch_vr);
        let sequence = asm.finalize_inline();

        harness.execute_sequence(&sequence);
        let out_vec = harness.read_output64(4);
        [out_vec[0], out_vec[1], out_vec[2], out_vec[3]]
    }

    fn run_fq_sub_inline(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
        let layout = InlineMemoryLayout::two_inputs(32, 32, 32);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();
        harness.load_input64(&a);
        harness.load_input2_64(&b);

        let mut asm =
            InstrAssembler::new_inline(0, false, harness.xlen(), &harness.cpu.vr_allocator);
        let a_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let b_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let out_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let scratch_vr: [VirtualRegisterGuard; 14] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let scratch = FqScratch::new(&scratch_vr);
        scratch.init_constants(&mut asm);

        for i in 0..4 {
            asm.emit_ld::<tracer::instruction::ld::LD>(*a_vr[i], INLINE_RS1, (i as i64) * 8);
            asm.emit_ld::<tracer::instruction::ld::LD>(*b_vr[i], INLINE_RS2, (i as i64) * 8);
        }

        fq_sub_mod(
            &mut asm,
            [*out_vr[0], *out_vr[1], *out_vr[2], *out_vr[3]],
            [*a_vr[0], *a_vr[1], *a_vr[2], *a_vr[3]],
            [*b_vr[0], *b_vr[1], *b_vr[2], *b_vr[3]],
            &scratch,
        );

        for i in 0..4 {
            asm.emit_s::<tracer::instruction::sd::SD>(INLINE_RS3, *out_vr[i], (i as i64) * 8);
        }

        drop(a_vr);
        drop(b_vr);
        drop(out_vr);
        drop(scratch_vr);
        let sequence = asm.finalize_inline();

        harness.execute_sequence(&sequence);
        let out_vec = harness.read_output64(4);
        [out_vec[0], out_vec[1], out_vec[2], out_vec[3]]
    }

    fn run_fq_neg_inline(a: [u64; 4]) -> [u64; 4] {
        let layout = InlineMemoryLayout::single_input(32, 32);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();
        // For `single_input`, rs1=output, rs2=input; we'll load input from rs2 mapping.
        harness.load_input64(&a);

        let mut asm =
            InstrAssembler::new_inline(0, false, harness.xlen(), &harness.cpu.vr_allocator);
        let a_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let out_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let scratch_vr: [VirtualRegisterGuard; 14] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let scratch = FqScratch::new(&scratch_vr);
        scratch.init_constants(&mut asm);

        for i in 0..4 {
            asm.emit_ld::<tracer::instruction::ld::LD>(*a_vr[i], INLINE_RS2, (i as i64) * 8);
        }

        fq_neg_mod(
            &mut asm,
            [*out_vr[0], *out_vr[1], *out_vr[2], *out_vr[3]],
            [*a_vr[0], *a_vr[1], *a_vr[2], *a_vr[3]],
            &scratch,
        );

        for i in 0..4 {
            asm.emit_s::<tracer::instruction::sd::SD>(INLINE_RS1, *out_vr[i], (i as i64) * 8);
        }

        drop(a_vr);
        drop(out_vr);
        drop(scratch_vr);
        let sequence = asm.finalize_inline();

        harness.execute_sequence(&sequence);
        let out_vec = harness.read_output64(4);
        [out_vec[0], out_vec[1], out_vec[2], out_vec[3]]
    }

    #[test]
    fn fq_add_sub_neg_match_arkworks_montgomery_limbs() {
        let mut rng = test_rng();
        for _ in 0..50 {
            let a = Fq::rand(&mut rng);
            let b = Fq::rand(&mut rng);
            assert_eq!(
                run_fq_add_inline(fq_to_limbs_mont(&a), fq_to_limbs_mont(&b)),
                fq_to_limbs_mont(&(a + b)),
                "Fq add mismatch"
            );
            assert_eq!(
                run_fq_sub_inline(fq_to_limbs_mont(&a), fq_to_limbs_mont(&b)),
                fq_to_limbs_mont(&(a - b)),
                "Fq sub mismatch"
            );
            assert_eq!(
                run_fq_neg_inline(fq_to_limbs_mont(&a)),
                fq_to_limbs_mont(&(-a)),
                "Fq neg mismatch"
            );
        }
    }

    fn run_fq2_mul_inline(a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
        let layout = InlineMemoryLayout::two_inputs(64, 64, 64);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();
        harness.load_input64(&a);
        harness.load_input2_64(&b);

        let mut asm =
            InstrAssembler::new_inline(0, false, harness.xlen(), &harness.cpu.vr_allocator);

        let a_vr: [VirtualRegisterGuard; 8] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let b_vr: [VirtualRegisterGuard; 8] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let out_vr: [VirtualRegisterGuard; 8] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let t0_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let t1_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let t2_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let scratch_vr: [VirtualRegisterGuard; 14] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let scratch = FqScratch::new(&scratch_vr);
        scratch.init_constants(&mut asm);

        for i in 0..8 {
            asm.emit_ld::<tracer::instruction::ld::LD>(*a_vr[i], INLINE_RS1, (i as i64) * 8);
            asm.emit_ld::<tracer::instruction::ld::LD>(*b_vr[i], INLINE_RS2, (i as i64) * 8);
        }

        let a_regs = Fq2Regs {
            c0: [*a_vr[0], *a_vr[1], *a_vr[2], *a_vr[3]],
            c1: [*a_vr[4], *a_vr[5], *a_vr[6], *a_vr[7]],
        };
        let b_regs = Fq2Regs {
            c0: [*b_vr[0], *b_vr[1], *b_vr[2], *b_vr[3]],
            c1: [*b_vr[4], *b_vr[5], *b_vr[6], *b_vr[7]],
        };
        let out_regs = Fq2Regs {
            c0: [*out_vr[0], *out_vr[1], *out_vr[2], *out_vr[3]],
            c1: [*out_vr[4], *out_vr[5], *out_vr[6], *out_vr[7]],
        };

        fq2_mul_karatsuba(
            &mut asm,
            out_regs,
            a_regs,
            b_regs,
            [*t0_vr[0], *t0_vr[1], *t0_vr[2], *t0_vr[3]],
            [*t1_vr[0], *t1_vr[1], *t1_vr[2], *t1_vr[3]],
            [*t2_vr[0], *t2_vr[1], *t2_vr[2], *t2_vr[3]],
            &scratch,
        );

        for i in 0..8 {
            asm.emit_s::<tracer::instruction::sd::SD>(INLINE_RS3, *out_vr[i], (i as i64) * 8);
        }

        drop(a_vr);
        drop(b_vr);
        drop(out_vr);
        drop(t0_vr);
        drop(t1_vr);
        drop(t2_vr);
        drop(scratch_vr);
        let sequence = asm.finalize_inline();

        harness.execute_sequence(&sequence);
        let out_vec = harness.read_output64(8);
        [
            out_vec[0], out_vec[1], out_vec[2], out_vec[3], out_vec[4], out_vec[5], out_vec[6],
            out_vec[7],
        ]
    }

    fn run_fq2_mul_by_xi_inline(a: [u64; 8]) -> [u64; 8] {
        let layout = InlineMemoryLayout::single_input(64, 64);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();
        harness.load_input64(&a);

        let mut asm =
            InstrAssembler::new_inline(0, false, harness.xlen(), &harness.cpu.vr_allocator);

        let fe_vr: [VirtualRegisterGuard; 8] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let t0_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let t1_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let scratch_vr: [VirtualRegisterGuard; 14] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let scratch = FqScratch::new(&scratch_vr);
        scratch.init_constants(&mut asm);

        for i in 0..8 {
            asm.emit_ld::<tracer::instruction::ld::LD>(*fe_vr[i], INLINE_RS2, (i as i64) * 8);
        }

        let fe_regs = Fq2Regs {
            c0: [*fe_vr[0], *fe_vr[1], *fe_vr[2], *fe_vr[3]],
            c1: [*fe_vr[4], *fe_vr[5], *fe_vr[6], *fe_vr[7]],
        };
        fq2_mul_by_xi_in_place(
            &mut asm,
            fe_regs,
            [*t0_vr[0], *t0_vr[1], *t0_vr[2], *t0_vr[3]],
            [*t1_vr[0], *t1_vr[1], *t1_vr[2], *t1_vr[3]],
            &scratch,
        );

        for i in 0..8 {
            asm.emit_s::<tracer::instruction::sd::SD>(INLINE_RS1, *fe_vr[i], (i as i64) * 8);
        }

        drop(fe_vr);
        drop(t0_vr);
        drop(t1_vr);
        drop(scratch_vr);
        let sequence = asm.finalize_inline();

        harness.execute_sequence(&sequence);
        let out_vec = harness.read_output64(8);
        [
            out_vec[0], out_vec[1], out_vec[2], out_vec[3], out_vec[4], out_vec[5], out_vec[6],
            out_vec[7],
        ]
    }

    #[test]
    fn fq2_mul_and_mul_by_xi_match_arkworks() {
        let mut rng = test_rng();
        for _ in 0..25 {
            let a = Fq2::rand(&mut rng);
            let b = Fq2::rand(&mut rng);
            assert_eq!(
                run_fq2_mul_inline(fq2_to_limbs_mont(&a), fq2_to_limbs_mont(&b)),
                fq2_to_limbs_mont(&(a * b)),
                "Fq2 mul mismatch"
            );
            let xi = <Fq6Config as Fp6Config>::NONRESIDUE;
            assert_eq!(
                run_fq2_mul_by_xi_inline(fq2_to_limbs_mont(&a)),
                fq2_to_limbs_mont(&(a * xi)),
                "Fq2 mul_by_xi mismatch"
            );
        }
    }

    fn run_fq6_mul_inline(a: [u64; 24], b: [u64; 24]) -> [u64; 24] {
        let layout = InlineMemoryLayout::two_inputs(192, 192, 384);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();
        harness.load_input64(&a);
        harness.load_input2_64(&b);

        let mut asm =
            InstrAssembler::new_inline(0, false, harness.xlen(), &harness.cpu.vr_allocator);

        let scratch_vr: [VirtualRegisterGuard; 14] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let scratch = FqScratch::new(&scratch_vr);
        scratch.init_constants(&mut asm);

        // Work regs for Fq2 ops.
        let a_vr: [VirtualRegisterGuard; 8] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let b_vr: [VirtualRegisterGuard; 8] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let out_vr: [VirtualRegisterGuard; 8] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let t0_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let t1_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let t2_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let w = Fq2Work {
            a: Fq2Regs {
                c0: [*a_vr[0], *a_vr[1], *a_vr[2], *a_vr[3]],
                c1: [*a_vr[4], *a_vr[5], *a_vr[6], *a_vr[7]],
            },
            b: Fq2Regs {
                c0: [*b_vr[0], *b_vr[1], *b_vr[2], *b_vr[3]],
                c1: [*b_vr[4], *b_vr[5], *b_vr[6], *b_vr[7]],
            },
            out: Fq2Regs {
                c0: [*out_vr[0], *out_vr[1], *out_vr[2], *out_vr[3]],
                c1: [*out_vr[4], *out_vr[5], *out_vr[6], *out_vr[7]],
            },
            t0: [*t0_vr[0], *t0_vr[1], *t0_vr[2], *t0_vr[3]],
            t1: [*t1_vr[0], *t1_vr[1], *t1_vr[2], *t1_vr[3]],
            t2: [*t2_vr[0], *t2_vr[1], *t2_vr[2], *t2_vr[3]],
        };

        fq6_mul_karatsuba_mem(
            &mut asm, INLINE_RS3, 0, INLINE_RS1, 0, INLINE_RS2, 0, w, &scratch,
        );

        drop(a_vr);
        drop(b_vr);
        drop(out_vr);
        drop(t0_vr);
        drop(t1_vr);
        drop(t2_vr);
        drop(scratch_vr);
        let sequence = asm.finalize_inline();

        harness.execute_sequence(&sequence);
        let out_vec = harness.read_output64(24);
        [
            out_vec[0],
            out_vec[1],
            out_vec[2],
            out_vec[3],
            out_vec[4],
            out_vec[5],
            out_vec[6],
            out_vec[7],
            out_vec[8],
            out_vec[9],
            out_vec[10],
            out_vec[11],
            out_vec[12],
            out_vec[13],
            out_vec[14],
            out_vec[15],
            out_vec[16],
            out_vec[17],
            out_vec[18],
            out_vec[19],
            out_vec[20],
            out_vec[21],
            out_vec[22],
            out_vec[23],
        ]
    }

    fn run_fq6_mul_by_v_inline(a: [u64; 24]) -> [u64; 24] {
        let layout = InlineMemoryLayout::single_input(192, 192);
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();
        harness.load_input64(&a);

        let mut asm =
            InstrAssembler::new_inline(0, false, harness.xlen(), &harness.cpu.vr_allocator);
        let scratch_vr: [VirtualRegisterGuard; 14] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let scratch = FqScratch::new(&scratch_vr);
        scratch.init_constants(&mut asm);

        let fe_vr: [VirtualRegisterGuard; 24] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        for i in 0..24 {
            asm.emit_ld::<tracer::instruction::ld::LD>(*fe_vr[i], INLINE_RS2, (i as i64) * 8);
        }

        let tmp_fq2_vr: [VirtualRegisterGuard; 8] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let t0_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());
        let t1_vr: [VirtualRegisterGuard; 4] =
            array::from_fn(|_| asm.allocator.allocate_for_inline());

        let fe = Fq6Regs {
            c0: Fq2Regs {
                c0: [*fe_vr[0], *fe_vr[1], *fe_vr[2], *fe_vr[3]],
                c1: [*fe_vr[4], *fe_vr[5], *fe_vr[6], *fe_vr[7]],
            },
            c1: Fq2Regs {
                c0: [*fe_vr[8], *fe_vr[9], *fe_vr[10], *fe_vr[11]],
                c1: [*fe_vr[12], *fe_vr[13], *fe_vr[14], *fe_vr[15]],
            },
            c2: Fq2Regs {
                c0: [*fe_vr[16], *fe_vr[17], *fe_vr[18], *fe_vr[19]],
                c1: [*fe_vr[20], *fe_vr[21], *fe_vr[22], *fe_vr[23]],
            },
        };
        let tmp_fq2 = Fq2Regs {
            c0: [
                *tmp_fq2_vr[0],
                *tmp_fq2_vr[1],
                *tmp_fq2_vr[2],
                *tmp_fq2_vr[3],
            ],
            c1: [
                *tmp_fq2_vr[4],
                *tmp_fq2_vr[5],
                *tmp_fq2_vr[6],
                *tmp_fq2_vr[7],
            ],
        };

        fq6_mul_by_nonresidue_v_in_place(
            &mut asm,
            fe,
            tmp_fq2,
            [*t0_vr[0], *t0_vr[1], *t0_vr[2], *t0_vr[3]],
            [*t1_vr[0], *t1_vr[1], *t1_vr[2], *t1_vr[3]],
            &scratch,
        );

        for i in 0..24 {
            asm.emit_s::<tracer::instruction::sd::SD>(INLINE_RS1, *fe_vr[i], (i as i64) * 8);
        }

        drop(fe_vr);
        drop(tmp_fq2_vr);
        drop(t0_vr);
        drop(t1_vr);
        drop(scratch_vr);
        let sequence = asm.finalize_inline();

        harness.execute_sequence(&sequence);
        let out_vec = harness.read_output64(24);
        [
            out_vec[0],
            out_vec[1],
            out_vec[2],
            out_vec[3],
            out_vec[4],
            out_vec[5],
            out_vec[6],
            out_vec[7],
            out_vec[8],
            out_vec[9],
            out_vec[10],
            out_vec[11],
            out_vec[12],
            out_vec[13],
            out_vec[14],
            out_vec[15],
            out_vec[16],
            out_vec[17],
            out_vec[18],
            out_vec[19],
            out_vec[20],
            out_vec[21],
            out_vec[22],
            out_vec[23],
        ]
    }

    #[test]
    fn fq6_mul_and_mul_by_v_match_arkworks() {
        let mut rng = test_rng();
        for _ in 0..10 {
            let a = Fq6::rand(&mut rng);
            let b = Fq6::rand(&mut rng);
            assert_eq!(
                run_fq6_mul_inline(fq6_to_limbs_mont(&a), fq6_to_limbs_mont(&b)),
                fq6_to_limbs_mont(&(a * b)),
                "Fq6 mul mismatch"
            );

            let v = Fq6::new(Fq2::ZERO, Fq2::ONE, Fq2::ZERO);
            assert_eq!(
                run_fq6_mul_by_v_inline(fq6_to_limbs_mont(&a)),
                fq6_to_limbs_mont(&(a * v)),
                "Fq6 mul_by_v mismatch"
            );
        }
    }

    fn run_gt_exp_inline(base: [u64; 48], exp: [u64; 4], layout: InlineMemoryLayout) -> [u64; 48] {
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();

        // rs1 = exp_ptr (4 limbs), rs2 = base_ptr (48 limbs), rs3 = out_ptr (48 limbs)
        harness.load_input64(&exp);
        harness.load_input2_64(&base);

        let asm = InstrAssembler::new_inline(0, false, harness.xlen(), &harness.cpu.vr_allocator);
        let sequence = bn254_gt_exp_sequence_builder(
            asm,
            FormatInline {
                rs1: INLINE_RS1,
                rs2: INLINE_RS2,
                rs3: INLINE_RS3,
            },
        );

        harness.execute_sequence(&sequence);
        let out_vec = harness.read_output64(48);
        let mut out = [0u64; 48];
        out.copy_from_slice(&out_vec);
        out
    }

    fn run_gt_sqr_inline(x: [u64; 48], layout: InlineMemoryLayout) -> [u64; 48] {
        let mut harness = InlineTestHarness::new(layout, Xlen::Bit64);
        harness.setup_registers();

        // rs1 = in_ptr (48 limbs), rs2 = ignored (we duplicate input), rs3 = out_ptr (48 limbs)
        harness.load_input64(&x);
        harness.load_input2_64(&x);

        let asm = InstrAssembler::new_inline(0, false, harness.xlen(), &harness.cpu.vr_allocator);
        let sequence = bn254_gt_sqr_sequence_builder(
            asm,
            FormatInline {
                rs1: INLINE_RS1,
                rs2: INLINE_RS2,
                rs3: INLINE_RS3,
            },
        );

        harness.execute_sequence(&sequence);
        let out_vec = harness.read_output64(48);
        let mut out = [0u64; 48];
        out.copy_from_slice(&out_vec);
        out
    }

    #[test]
    fn gt_sqr_pairing_output_matches_arkworks_square() {
        let mut rng = test_rng();
        for _ in 0..4 {
            // Pairing outputs are in BN254 GT ⊂ cyclotomic subgroup, so cyclotomic square must match `square()`.
            let g1 = G1Projective::rand(&mut rng);
            let g2 = G2Projective::rand(&mut rng);
            let gt: Fq12 = Bn254::pairing(g1, g2).0;

            let expected = fq12_to_limbs_mont(&gt.square());
            let got = run_gt_sqr_inline(
                fq12_to_limbs_mont(&gt),
                InlineMemoryLayout::two_inputs(384, 384, 384),
            );
            assert_eq!(got, expected, "GT sqr mismatch vs arkworks Fq12::square()");
        }
    }

    #[test]
    fn gt_exp_exp2_matches_arkworks_fq12_square() {
        let mut rng = test_rng();
        for _ in 0..2 {
            let a = Fq12::rand(&mut rng);
            let expected = fq12_to_limbs_mont(&a.square());
            let got = run_gt_exp_inline(
                fq12_to_limbs_mont(&a),
                [2u64, 0, 0, 0],
                InlineMemoryLayout::two_inputs(32, 384, 384),
            );
            assert_eq!(
                got, expected,
                "GT exp (exp=2) mismatch vs arkworks Fq12::square()"
            );
        }
    }

    #[test]
    fn gt_exp_random_matches_arkworks_pow() {
        use ark_bn254::Fr;
        use ark_ff::PrimeField;

        let mut rng = test_rng();
        // Keep this small; the full 256-bit fixed schedule is expensive.
        for _ in 0..1 {
            let a = Fq12::rand(&mut rng);
            let e = Fr::rand(&mut rng);
            let exp_limbs: [u64; 4] = e.into_bigint().0;

            let expected = fq12_to_limbs_mont(&a.pow(exp_limbs));
            let got = run_gt_exp_inline(
                fq12_to_limbs_mont(&a),
                exp_limbs,
                InlineMemoryLayout::two_inputs(32, 384, 384),
            );
            assert_eq!(got, expected, "GT exp mismatch vs arkworks Fq12::pow()");
        }
    }

    #[test]
    #[should_panic]
    fn gt_exp_exp2_rejects_out_eq_base_aliasing() {
        use tracer::utils::inline_test_harness::RegisterMapping;

        let mut rng = test_rng();
        let a = Fq12::rand(&mut rng);

        // Force rs3 (out_ptr) to alias rs2 (base_ptr) by pointing both at Input2.
        let layout = InlineMemoryLayout {
            input_base: tracer::emulator::mmu::DRAM_BASE,
            input_size: 32,
            input2_base: Some(tracer::emulator::mmu::DRAM_BASE + 32),
            input2_size: Some(384),
            output_base: tracer::emulator::mmu::DRAM_BASE + 32,
            output_size: 384,
            rs1_mapping: RegisterMapping::Input,
            rs2_mapping: RegisterMapping::Input2,
            rs3_mapping: Some(RegisterMapping::Input2),
        };

        // Should panic due to the explicit `base_ptr != out_ptr` assertion.
        let _ = run_gt_exp_inline(fq12_to_limbs_mont(&a), [2u64, 0, 0, 0], layout);
    }
}
