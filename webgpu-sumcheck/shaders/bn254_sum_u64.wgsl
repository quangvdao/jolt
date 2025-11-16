// BN254 field arithmetic on WebGPU in Montgomery form using 4×64-bit limbs.
// Representation: 4 little-endian 64-bit limbs, base 2^64, storing Montgomery residues.
// All operations are performed modulo the BN254 scalar field modulus using Montgomery reduction.

const FR_NUM_LIMBS : u32 = 4u; // 4 × 64-bit limbs = 256 bits
const WORKGROUP_SIZE : u32 = 256u;

// Each field element is 4×u64, stored as a vec4<u64> for alignment.
struct Fr {
    limbs: vec4<u64>, // limbs[0..3], least-significant first
};

struct Buffer {
    data: array<Fr>,
};

struct Params {
    len: u32,
    phase: u32,
};

@group(0) @binding(0)
var<storage, read> p_buf: Buffer;

@group(0) @binding(1)
var<storage, read> q_buf: Buffer;

@group(0) @binding(2)
var<storage, read_write> out_buf: Buffer;

@group(0) @binding(3)
var<uniform> params: Params;

var<workgroup> shared_sums: array<Fr, WORKGROUP_SIZE>;

// Sumcheck-specific bindings and parameters (mirrors 32-bit BN254 shader).
struct SumcheckParams {
    len: u32,
    _pad0: u32,
    r: Fr,
};

@group(0) @binding(0)
var<storage, read> sc_p_in: Buffer;

@group(0) @binding(1)
var<storage, read_write> sc_p_out: Buffer;

@group(0) @binding(2)
var<storage, read> sc_q_in: Buffer;

@group(0) @binding(3)
var<storage, read_write> sc_q_out: Buffer;

@group(0) @binding(4)
var<storage, read_write> sc_g0_partial: Buffer;

@group(0) @binding(5)
var<storage, read_write> sc_g1_partial: Buffer;

@group(0) @binding(6)
var<storage, read_write> sc_g2_partial: Buffer;

@group(0) @binding(7)
var<uniform> sc_params: SumcheckParams;

var<workgroup> sc_shared_g0: array<Fr, WORKGROUP_SIZE>;
var<workgroup> sc_shared_g1: array<Fr, WORKGROUP_SIZE>;
var<workgroup> sc_shared_g2: array<Fr, WORKGROUP_SIZE>;

fn fr_zero() -> Fr {
    return Fr(vec4<u64>(0u, 0u, 0u, 0u));
}

// BN254 scalar field modulus in base-2^64, little-endian, represented via
// 32-bit chunks so that all numeric literals fit within 32 bits.
// Original 32-bit limbs (little-endian, from the 8×u32 representation):
// [4026531841, 1138881939, 2042196113, 674490440,
//  2172737629, 3092268470, 3778125865, 811880050]
// Recombined into 4×u64 limbs as:
// [0x43e1f593f0000001, 0x2833e84879b97091,
//  0xb85045b68181585d, 0x30644e72e131a029]
fn fr_modulus() -> Fr {
    let limb0: u64 = (u64(1138881939u) << 32u) | u64(4026531841u);
    let limb1: u64 = (u64(674490440u) << 32u) | u64(2042196113u);
    let limb2: u64 = (u64(3092268470u) << 32u) | u64(2172737629u);
    let limb3: u64 = (u64(811880050u) << 32u) | u64(3778125865u);
    return Fr(vec4<u64>(limb0, limb1, limb2, limb3));
}

// -p^{-1} mod 2^64 for the BN254 scalar field modulus (first 64-bit limb).
// Computed as 0xc2e1f593efffffff, satisfying:
//   fr_modulus().limbs[0] * fr_inv64() ≡ -1 (mod 2^64).
fn fr_inv64() -> u64 {
    let hi: u64 = u64(0xc2e1f593u);
    let lo: u64 = u64(0xefffffffu);
    return (hi << 32u) | lo;
}

struct Mul64Result {
    lo: u64,
    hi: u64,
};

struct Add64Result {
    sum: u64,
    carry: u64,
};

fn add_with_carry64(a: u64, b: u64, carry_in: u64) -> Add64Result {
    let sum = a + b;
    let carry0 = select(0u, 1u, sum < a);

    let sum2 = sum + carry_in;
    let carry1 = select(0u, 1u, sum2 < sum);

    var result: Add64Result;
    result.sum = sum2;
    result.carry = carry0 | carry1;
    return result;
}

// 64×64 → 128 multiplication using 32-bit chunks, staying within u64 arithmetic.
fn mul64_emulated(a: u64, b: u64) -> Mul64Result {
    let MASK32: u64 = 0xFFFFFFFFu;

    let a0 = a & MASK32;
    let a1 = a >> 32u;
    let b0 = b & MASK32;
    let b1 = b >> 32u;

    let c0 = a0 * b0; // < 2^64

    let c1a = a0 * b1; // < 2^64
    let c1b = a1 * b0; // < 2^64
    let tmp_c1 = add_with_carry64(c1a, c1b, 0u);
    let c1_low = tmp_c1.sum;
    let c1_high_extra = tmp_c1.carry; // 0 or 1

    let c2 = a1 * b1; // < 2^64

    let m1_low32 = c1_low & MASK32;
    let m1_high_total = (c1_low >> 32u) + (c1_high_extra << 32u);

    let low_part = m1_low32 << 32u;
    let tmp_low = add_with_carry64(c0, low_part, 0u);
    let low_tmp = tmp_low.sum;
    let carry0 = tmp_low.carry; // 0 or 1

    var res: Mul64Result;
    res.lo = low_tmp;
    res.hi = c2 + m1_high_total + carry0;
    return res;
}

// Limb helpers for Fr

fn fr_get_limb(a: Fr, idx: u32) -> u64 {
    return a.limbs[idx];
}

fn fr_set_limb(a: ptr<function, Fr>, idx: u32, value: u64) {
    (*a).limbs[idx] = value;
}

fn fr_to_limbs(a: Fr) -> array<u64, 4u> {
    var limbs: array<u64, 4u>;
    var i: u32 = 0u;
    loop {
        if (i >= FR_NUM_LIMBS) {
            break;
        }
        limbs[i] = fr_get_limb(a, i);
        i = i + 1u;
    }
    return limbs;
}

fn fr_from_limbs(limbs: ptr<function, array<u64, 4u>>) -> Fr {
    var r: Fr = fr_zero();
    var i: u32 = 0u;
    loop {
        if (i >= FR_NUM_LIMBS) {
            break;
        }
        fr_set_limb(&r, i, (*limbs)[i]);
        i = i + 1u;
    }
    return r;
}

fn fr_add_raw(a: Fr, b: Fr) -> Fr {
    var r: Fr = fr_zero();
    var tmp: Add64Result;
    var carry: u64 = 0u;
    var i: u32 = 0u;
    loop {
        if (i >= FR_NUM_LIMBS) {
            break;
        }
        let ai = fr_get_limb(a, i);
        let bi = fr_get_limb(b, i);
        tmp = add_with_carry64(ai, bi, carry);
        fr_set_limb(&r, i, tmp.sum);
        carry = tmp.carry;
        i = i + 1u;
    }
    return r;
}

fn limbs_geq_modulus(limbs: ptr<function, array<u64, 4u>>) -> bool {
    // Compare limbs >= FR_MODULUS, treating limbs as little-endian.
    let modulus = fr_modulus();
    var i: i32 = i32(FR_NUM_LIMBS) - 1;
    loop {
        if (i < 0) {
            break;
        }
        let idx: u32 = u32(i);
        let ai = (*limbs)[idx];
        let mi = fr_get_limb(modulus, idx);
        if (ai > mi) {
            return true;
        }
        if (ai < mi) {
            return false;
        }
        i = i - 1;
    }
    // Equal
    return true;
}

fn limbs_sub_modulus(a: ptr<function, array<u64, 4u>>) {
    let modulus = fr_modulus();
    var borrow: u64 = 0u;
    var i: u32 = 0u;
    loop {
        if (i >= FR_NUM_LIMBS) {
            break;
        }
        let ai = (*a)[i];
        let mi = fr_get_limb(modulus, i);
        let sub = mi + borrow;
        let diff = ai - sub;
        borrow = select(0u, 1u, ai < sub);
        (*a)[i] = diff;
        i = i + 1u;
    }
}

fn fr_geq_modulus(a: Fr) -> bool {
    // Compare a >= FR_MODULUS, treating limbs as little-endian.
    let modulus = fr_modulus();
    var i: i32 = i32(FR_NUM_LIMBS) - 1;
    loop {
        if (i < 0) {
            break;
        }
        let idx: u32 = u32(i);
        let ai = fr_get_limb(a, idx);
        let mi = fr_get_limb(modulus, idx);
        if (ai > mi) {
            return true;
        }
        if (ai < mi) {
            return false;
        }
        i = i - 1;
    }
    // Equal
    return true;
}

fn fr_conditional_sub_modulus(a: Fr) -> Fr {
    if (fr_geq_modulus(a)) {
        return fr_sub_raw(a, fr_modulus());
    }
    return a;
}

fn fr_add(a: Fr, b: Fr) -> Fr {
    let raw = fr_add_raw(a, b);
    return fr_conditional_sub_modulus(raw);
}

fn fr_sub_raw(a: Fr, b: Fr) -> Fr {
    var r: Fr = fr_zero();
    var borrow: u64 = 0u;
    var i: u32 = 0u;
    loop {
        if (i >= FR_NUM_LIMBS) {
            break;
        }
        let ai = fr_get_limb(a, i);
        let bi = fr_get_limb(b, i);
        let sub = bi + borrow;
        let diff = ai - sub;
        borrow = select(0u, 1u, ai < sub);
        fr_set_limb(&r, i, diff);
        i = i + 1u;
    }
    return r;
}

fn fr_sub(a: Fr, b: Fr) -> Fr {
    var r = fr_sub_raw(a, b);
    // If we borrowed, result is negative; add modulus back.
    // Detect borrow by checking if a < b.
    if (!fr_geq(a, b)) {
        r = fr_add_raw(r, fr_modulus());
    }
    return fr_conditional_sub_modulus(r);
}

fn fr_geq(a: Fr, b: Fr) -> bool {
    var i: i32 = i32(FR_NUM_LIMBS) - 1;
    loop {
        if (i < 0) {
            break;
        }
        let idx: u32 = u32(i);
        let ai = fr_get_limb(a, idx);
        let bi = fr_get_limb(b, idx);
        if (ai > bi) {
            return true;
        }
        if (ai < bi) {
            return false;
        }
        i = i - 1;
    }
    return true;
}

fn fr_mul(a: Fr, b: Fr) -> Fr {
    // Montgomery multiplication in base 2^64 with 4 limbs (CIOS), operating on
    // Montgomery residues. Structure mirrors the 32-bit-limb implementation.
    var t: array<u64, 5u>;
    // Initialize t to zero.
    var k_init: u32 = 0u;
    loop {
        if (k_init >= 5u) {
            break;
        }
        t[k_init] = 0u;
        k_init = k_init + 1u;
    }

    var i: u32 = 0u;
    loop {
        if (i >= FR_NUM_LIMBS) {
            break;
        }

        let bi = fr_get_limb(b, i);

        // t += a * bi
        var carry: u64 = 0u;
        var j: u32 = 0u;
        loop {
            if (j >= FR_NUM_LIMBS) {
                break;
            }
            let aj = fr_get_limb(a, j);
            let prod = mul64_emulated(aj, bi);

            let tmp1 = add_with_carry64(t[j], prod.lo, 0u);
            let tmp2 = add_with_carry64(tmp1.sum, carry, 0u);
            let c0 = tmp1.carry + tmp2.carry;

            t[j] = tmp2.sum;
            carry = prod.hi + c0;

            j = j + 1u;
        }
        t[4u] = t[4u] + carry;

        // m_i = t[0] * (-p^{-1} mod 2^64)
        let m_i = t[0u] * fr_inv64();

        // t += m_i * p
        var carry2: u64 = 0u;
        var j2: u32 = 0u;
        loop {
            if (j2 >= FR_NUM_LIMBS) {
                break;
            }
            let mj = fr_get_limb(fr_modulus(), j2);
            let prod2 = mul64_emulated(m_i, mj);

            let tmp1b = add_with_carry64(t[j2], prod2.lo, 0u);
            let tmp2b = add_with_carry64(tmp1b.sum, carry2, 0u);
            let c0b = tmp1b.carry + tmp2b.carry;

            t[j2] = tmp2b.sum;
            carry2 = prod2.hi + c0b;

            j2 = j2 + 1u;
        }
        t[4u] = t[4u] + carry2;

        // t = (t + m_i * p) / base: shift one limb to the right.
        var k: u32 = 0u;
        loop {
            if (k >= FR_NUM_LIMBS) {
                break;
            }
            t[k] = t[k + 1u];
            k = k + 1u;
        }
        t[4u] = 0u;

        i = i + 1u;
    }

    // Extract N limbs and do the final conditional subtraction.
    var out_limbs: array<u64, 4u>;
    var i_out: u32 = 0u;
    loop {
        if (i_out >= FR_NUM_LIMBS) {
            break;
        }
        out_limbs[i_out] = t[i_out];
        i_out = i_out + 1u;
    }
    if (limbs_geq_modulus(&out_limbs)) {
        limbs_sub_modulus(&out_limbs);
    }
    return fr_from_limbs(&out_limbs);
}

// ------------------------------------------------------------------------
// Inner-product kernel over BN254 using 4×u64 Fr arithmetic.
// ------------------------------------------------------------------------

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let idx = global_id.x;
    let local = local_id.x;

    var acc: Fr = fr_zero();
    if (idx < params.len) {
        if (params.phase == 0u) {
            let a = p_buf.data[idx];
            let b = q_buf.data[idx];
            acc = fr_mul(a, b);
        } else {
            acc = p_buf.data[idx];
        }
    }

    shared_sums[local] = acc;
    workgroupBarrier();

    var stride: u32 = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }

        if (local < stride) {
            shared_sums[local] = fr_add(shared_sums[local], shared_sums[local + stride]);
        }

        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local == 0u) {
        out_buf.data[workgroup_id.x] = shared_sums[0u];
    }
}

// ------------------------------------------------------------------------
// Sumcheck kernels over BN254 (4×u64 representation).
// ------------------------------------------------------------------------

// First kernel: compute per-workgroup partial sums of A, B, C coefficients.
@compute @workgroup_size(WORKGROUP_SIZE)
fn sumcheck_eval_round(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let local = local_id.x;
    let len = sc_params.len;
    let pair_idx = global_id.x;

    var acc0: Fr = fr_zero(); // sum of A coefficients
    var acc1: Fr = fr_zero(); // sum of B coefficients
    var acc2: Fr = fr_zero(); // sum of C coefficients

    let base = 2u * pair_idx;
    if (base + 1u < len) {
        let p0 = sc_p_in.data[base];
        let p1 = sc_p_in.data[base + 1u];
        let q0 = sc_q_in.data[base];
        let q1 = sc_q_in.data[base + 1u];

        let dp = fr_sub(p1, p0);
        let dq = fr_sub(q1, q0);

        let a = fr_mul(p0, q0);
        let b1 = fr_mul(p0, dq);
        let b2 = fr_mul(q0, dp);
        let b = fr_add(b1, b2);
        let c = fr_mul(dp, dq);

        acc0 = fr_add(acc0, a);
        acc1 = fr_add(acc1, b);
        acc2 = fr_add(acc2, c);
    }

    sc_shared_g0[local] = acc0;
    sc_shared_g1[local] = acc1;
    sc_shared_g2[local] = acc2;
    workgroupBarrier();

    var stride: u32 = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }

        if (local < stride) {
            sc_shared_g0[local] =
                fr_add(sc_shared_g0[local], sc_shared_g0[local + stride]);
            sc_shared_g1[local] =
                fr_add(sc_shared_g1[local], sc_shared_g1[local + stride]);
            sc_shared_g2[local] =
                fr_add(sc_shared_g2[local], sc_shared_g2[local + stride]);
        }

        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local == 0u) {
        sc_g0_partial.data[workgroup_id.x] = sc_shared_g0[0u];
        sc_g1_partial.data[workgroup_id.x] = sc_shared_g1[0u];
        sc_g2_partial.data[workgroup_id.x] = sc_shared_g2[0u];
    }
}

// Reduction kernel for the univariate coefficients:
//  - Input:  sc_g0_partial[0..len), sc_g1_partial[0..len), sc_g2_partial[0..len)
//  - Output: per-workgroup partial sums written back into the same buffers at
//            indices [0..num_workgroups). Repeated application reduces the
//            arrays down to length 1.
@compute @workgroup_size(WORKGROUP_SIZE)
fn sumcheck_reduce_coeffs(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let local = local_id.x;
    let len = sc_params.len;
    let idx = global_id.x;

    var acc0: Fr = fr_zero();
    var acc1: Fr = fr_zero();
    var acc2: Fr = fr_zero();

    if (idx < len) {
        acc0 = sc_g0_partial.data[idx];
        acc1 = sc_g1_partial.data[idx];
        acc2 = sc_g2_partial.data[idx];
    }

    sc_shared_g0[local] = acc0;
    sc_shared_g1[local] = acc1;
    sc_shared_g2[local] = acc2;
    workgroupBarrier();

    var stride: u32 = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }

        if (local < stride) {
            sc_shared_g0[local] =
                fr_add(sc_shared_g0[local], sc_shared_g0[local + stride]);
            sc_shared_g1[local] =
                fr_add(sc_shared_g1[local], sc_shared_g1[local + stride]);
            sc_shared_g2[local] =
                fr_add(sc_shared_g2[local], sc_shared_g2[local + stride]);
        }

        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local == 0u) {
        sc_g0_partial.data[workgroup_id.x] = sc_shared_g0[0u];
        sc_g1_partial.data[workgroup_id.x] = sc_shared_g1[0u];
        sc_g2_partial.data[workgroup_id.x] = sc_shared_g2[0u];
    }
}

// Second kernel: apply a verifier challenge r to bind one variable.
@compute @workgroup_size(WORKGROUP_SIZE)
fn sumcheck_bind_round(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let len = sc_params.len;
    let pair_idx = global_id.x;
    let base = 2u * pair_idx;

    if (base + 1u >= len) {
        return;
    }

    let p0 = sc_p_in.data[base];
    let p1 = sc_p_in.data[base + 1u];
    let q0 = sc_q_in.data[base];
    let q1 = sc_q_in.data[base + 1u];

    let dp = fr_sub(p1, p0);
    let dq = fr_sub(q1, q0);
    let r = sc_params.r;

    let rp = fr_mul(r, dp);
    let rq = fr_mul(r, dq);

    let p_next = fr_add(p0, rp);
    let q_next = fr_add(q0, rq);

    sc_p_out.data[pair_idx] = p_next;
    sc_q_out.data[pair_idx] = q_next;
}


