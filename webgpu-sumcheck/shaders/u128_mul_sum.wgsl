const NUM_LIMBS : u32 = 4u;
const WORKGROUP_SIZE : u32 = 256u;

alias U128 = vec4<u32>;

struct Buffer {
    data: array<U128>,
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

var<workgroup> shared_sums: array<U128, WORKGROUP_SIZE>;

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


