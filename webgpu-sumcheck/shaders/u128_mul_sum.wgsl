const NUM_LIMBS : u32 = 4u;
const WORKGROUP_SIZE : u32 = 256u;

alias U128 = vec4<u32>;

struct Buffer {
    data: array<U128>,
};

// Parameters for the simple inner-product kernel
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

var<workgroup> shared_sums: array<U128, WORKGROUP_SIZE>;

// ------------------------------------------------------------------------
// Sumcheck-specific bindings and parameters for f = p * q where
// p and q are multilinear in each variable. The product f has
// degree 2 in each variable, so we must track p and q separately
// and the univariate message g_j(X) has degree <= 2.
// ------------------------------------------------------------------------

struct SumcheckParams {
    len: u32,
    _pad0: u32,
    r: U128,
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

var<workgroup> sc_shared_g0: array<U128, WORKGROUP_SIZE>;
var<workgroup> sc_shared_g1: array<U128, WORKGROUP_SIZE>;
var<workgroup> sc_shared_g2: array<U128, WORKGROUP_SIZE>;

fn zero_u128() -> U128 {
    return vec4<u32>(0u, 0u, 0u, 0u);
}

struct AddResult {
    sum: u32,
    carry: u32,
};

fn add_with_carry(a: u32, b: u32, carry_in: u32) -> AddResult {
    // First add a and b
    let sum = a + b;
    let carry0 = select(0u, 1u, sum < a);

    // Then add the incoming carry
    let sum2 = sum + carry_in;
    let carry1 = select(0u, 1u, sum2 < sum);

    var result: AddResult;
    result.sum = sum2;
    result.carry = carry0 | carry1;
    return result;
}

struct MulWideResult {
    lo: u32,
    hi: u32,
};

fn mul_wide(a: u32, b: u32) -> MulWideResult {
    let a_lo = a & 0xFFFFu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xFFFFu;
    let b_hi = b >> 16u;

    let p0 = a_lo * b_lo;
    let p1 = a_lo * b_hi;
    let p2 = a_hi * b_lo;
    let p3 = a_hi * b_hi;

    let carry = p0 >> 16u;
    let mid = carry + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);

    let lo = (p0 & 0xFFFFu) | ((mid & 0xFFFFu) << 16u);
    let hi = (mid >> 16u) + (p1 >> 16u) + (p2 >> 16u) + p3;

    var res: MulWideResult;
    res.lo = lo;
    res.hi = hi;
    return res;
}

// Add `value` into limb `idx` of `c`, propagating carries to higher limbs as needed.
fn add_into(c: ptr<function, U128>, idx: u32, value: u32) {
    if (idx >= NUM_LIMBS) {
        return;
    }

    var tmp: AddResult;
    var carry: u32 = 0u;

    if (idx == 0u) {
        tmp = add_with_carry((*c).x, value, carry);
        (*c).x = tmp.sum;
        carry = tmp.carry;

        tmp = add_with_carry((*c).y, 0u, carry);
        (*c).y = tmp.sum;
        carry = tmp.carry;

        tmp = add_with_carry((*c).z, 0u, carry);
        (*c).z = tmp.sum;
        carry = tmp.carry;

        tmp = add_with_carry((*c).w, 0u, carry);
        (*c).w = tmp.sum;
    } else if (idx == 1u) {
        tmp = add_with_carry((*c).y, value, carry);
        (*c).y = tmp.sum;
        carry = tmp.carry;

        tmp = add_with_carry((*c).z, 0u, carry);
        (*c).z = tmp.sum;
        carry = tmp.carry;

        tmp = add_with_carry((*c).w, 0u, carry);
        (*c).w = tmp.sum;
    } else if (idx == 2u) {
        tmp = add_with_carry((*c).z, value, carry);
        (*c).z = tmp.sum;
        carry = tmp.carry;

        tmp = add_with_carry((*c).w, 0u, carry);
        (*c).w = tmp.sum;
    } else { // idx == 3
        tmp = add_with_carry((*c).w, value, carry);
        (*c).w = tmp.sum;
    }
}

// 128-bit multiplication modulo 2^128.
fn mul_u128(a: U128, b: U128) -> U128 {
    var acc: U128 = zero_u128();
    var mw: MulWideResult;

    // i = 0
    mw = mul_wide(a.x, b.x);
    add_into(&acc, 0u, mw.lo);
    add_into(&acc, 1u, mw.hi);

    mw = mul_wide(a.x, b.y);
    add_into(&acc, 1u, mw.lo);
    add_into(&acc, 2u, mw.hi);

    mw = mul_wide(a.x, b.z);
    add_into(&acc, 2u, mw.lo);
    add_into(&acc, 3u, mw.hi);

    mw = mul_wide(a.x, b.w);
    add_into(&acc, 3u, mw.lo);
    // high part would go to limb 4, discarded mod 2^128

    // i = 1
    mw = mul_wide(a.y, b.x);
    add_into(&acc, 1u, mw.lo);
    add_into(&acc, 2u, mw.hi);

    mw = mul_wide(a.y, b.y);
    add_into(&acc, 2u, mw.lo);
    add_into(&acc, 3u, mw.hi);

    mw = mul_wide(a.y, b.z);
    add_into(&acc, 3u, mw.lo);
    // high part goes to limb 4, discarded

    mw = mul_wide(a.y, b.w);
    // contributes only above limb 3, ignored

    // i = 2
    mw = mul_wide(a.z, b.x);
    add_into(&acc, 2u, mw.lo);
    add_into(&acc, 3u, mw.hi);

    mw = mul_wide(a.z, b.y);
    add_into(&acc, 3u, mw.lo);
    // high part ignored

    mw = mul_wide(a.z, b.z);
    // all contributions start at limb 4, ignored

    mw = mul_wide(a.z, b.w);
    // ignored

    // i = 3
    mw = mul_wide(a.w, b.x);
    add_into(&acc, 3u, mw.lo);
    // high part ignored

    mw = mul_wide(a.w, b.y);
    // ignored

    mw = mul_wide(a.w, b.z);
    // ignored

    mw = mul_wide(a.w, b.w);
    // ignored

    return acc;
}

// 128-bit addition modulo 2^128.
fn add_u128(a: U128, b: U128) -> U128 {
    var r: U128;
    var tmp: AddResult;
    var carry: u32 = 0u;

    tmp = add_with_carry(a.x, b.x, carry);
    r.x = tmp.sum;
    carry = tmp.carry;

    tmp = add_with_carry(a.y, b.y, carry);
    r.y = tmp.sum;
    carry = tmp.carry;

    tmp = add_with_carry(a.z, b.z, carry);
    r.z = tmp.sum;
    carry = tmp.carry;

    tmp = add_with_carry(a.w, b.w, carry);
    r.w = tmp.sum;
    // Final carry is discarded (addition mod 2^128)

    return r;
}

// 128-bit subtraction modulo 2^128: returns a - b mod 2^128.
fn sub_u128(a: U128, b: U128) -> U128 {
    var r: U128;
    var borrow: u32 = 0u;

    // x-limb
    let bx = b.x + borrow;
    let diff_x = a.x - bx;
    borrow = select(0u, 1u, a.x < bx);
    r.x = diff_x;

    // y-limb
    let by = b.y + borrow;
    let diff_y = a.y - by;
    borrow = select(0u, 1u, a.y < by);
    r.y = diff_y;

    // z-limb
    let bz = b.z + borrow;
    let diff_z = a.z - bz;
    borrow = select(0u, 1u, a.z < bz);
    r.z = diff_z;

    // w-limb
    let bw = b.w + borrow;
    let diff_w = a.w - bw;
    // final borrow is discarded (subtraction mod 2^128)
    r.w = diff_w;

    return r;
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let idx = global_id.x;
    let local = local_id.x;

    var acc: U128 = zero_u128();
    if (idx < params.len) {
        if (params.phase == 0u) {
            let a = p_buf.data[idx];
            let b = q_buf.data[idx];
            acc = mul_u128(a, b);
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
            shared_sums[local] = add_u128(shared_sums[local], shared_sums[local + stride]);
        }

        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local == 0u) {
        out_buf.data[workgroup_id.x] = shared_sums[0u];
    }
}

// ------------------------------------------------------------------------
// Sumcheck kernels
// ------------------------------------------------------------------------

// First kernel for a sumcheck round for f = p * q:
//  - Input:  sc_p_in[0..len), sc_q_in[0..len)
//  - For each pair (p0, p1), (q0, q1) corresponding to bit j ∈ {0,1}, we have
//        p_j(X) = p0 + X (p1 - p0)
//        q_j(X) = q0 + X (q1 - q0)
//    and hence
//        p_j(X) * q_j(X) = A + B X + C X^2
//    where
//        A = p0 * q0
//        B = p0 * (q1 - q0) + q0 * (p1 - p0)
//        C = (p1 - p0) * (q1 - q0)
//  - This kernel computes per-workgroup partial sums of A, B, C over all pairs,
//    writing them into sc_g0_partial, sc_g1_partial, sc_g2_partial.
@compute @workgroup_size(WORKGROUP_SIZE)
fn sumcheck_eval_round(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let local = local_id.x;
    let len = sc_params.len;
    let pair_idx = global_id.x;

    var acc0: U128 = zero_u128(); // sum of A coefficients
    var acc1: U128 = zero_u128(); // sum of B coefficients
    var acc2: U128 = zero_u128(); // sum of C coefficients

    // Each thread handles one (p0, p1), (q0, q1) pair if it is in range.
    let base = 2u * pair_idx;
    if (base + 1u < len) {
        let p0 = sc_p_in.data[base];
        let p1 = sc_p_in.data[base + 1u];
        let q0 = sc_q_in.data[base];
        let q1 = sc_q_in.data[base + 1u];

        let dp = sub_u128(p1, p0);
        let dq = sub_u128(q1, q0);

        let a = mul_u128(p0, q0);
        let b1 = mul_u128(p0, dq);
        let b2 = mul_u128(q0, dp);
        let b = add_u128(b1, b2);
        let c = mul_u128(dp, dq);

        acc0 = add_u128(acc0, a);
        acc1 = add_u128(acc1, b);
        acc2 = add_u128(acc2, c);
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
                add_u128(sc_shared_g0[local], sc_shared_g0[local + stride]);
            sc_shared_g1[local] =
                add_u128(sc_shared_g1[local], sc_shared_g1[local + stride]);
            sc_shared_g2[local] =
                add_u128(sc_shared_g2[local], sc_shared_g2[local + stride]);
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

// Second kernel for a sumcheck round:
//  - Input:  sc_p_in[0..len), sc_q_in[0..len)
//  - Uniform: sc_params.r is the challenge r
//  - Output:
//        sc_p_out[i] = p0 + r * (p1 - p0)
//        sc_q_out[i] = q0 + r * (q1 - q0)
//    for p0 = sc_p_in[2*i], p1 = sc_p_in[2*i+1],
//        q0 = sc_q_in[2*i], q1 = sc_q_in[2*i+1].
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

    let dp = sub_u128(p1, p0);
    let dq = sub_u128(q1, q0);
    let r = sc_params.r;

    let rp = mul_u128(r, dp);
    let rq = mul_u128(r, dq);

    let p_next = add_u128(p0, rp);
    let q_next = add_u128(q0, rq);

    sc_p_out.data[pair_idx] = p_next;
    sc_q_out.data[pair_idx] = q_next;
}


