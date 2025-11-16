use ark_bn254::Fr;
use ark_ff::{PrimeField, UniformRand};

const N_LIMBS: usize = 8; // 8 × 32-bit limbs

#[derive(Clone, Copy, Debug)]
struct Fr32 {
    limbs: [u32; N_LIMBS], // little-endian 32-bit limbs of Montgomery residue
}

fn fr_to_fr32_mont(x: Fr) -> Fr32 {
    // Arkworks Fr stores its internal representation in Montgomery form in the
    // public BigInt field (type BigInt<4>, 64-bit limbs).
    type FrBigInt = <Fr as PrimeField>::BigInt;
    let mont: &FrBigInt = &x.0;
    let mut limbs32 = [0u32; N_LIMBS];
    for (i, limb64) in mont.0.iter().enumerate() {
        let lo = (*limb64 & 0xFFFF_FFFF) as u32;
        let hi = (*limb64 >> 32) as u32;
        limbs32[2 * i] = lo;
        limbs32[2 * i + 1] = hi;
    }
    Fr32 { limbs: limbs32 }
}

fn fr32_mont_to_fr(x: &Fr32) -> Fr {
    type FrBigInt = <Fr as PrimeField>::BigInt;
    let mut raw64 = [0u64; 4];
    for i in 0..4 {
        let lo = x.limbs[2 * i] as u64;
        let hi = (x.limbs[2 * i + 1] as u64) << 32;
        raw64[i] = lo | hi;
    }
    let big = FrBigInt::new(raw64);
    // Interpret the BigInt as an internal Montgomery residue, without
    // additional reduction/conversion.
    Fr::from_bigint_unchecked(big).expect("from_bigint_unchecked should not fail for Fr")
}

#[derive(Clone, Copy)]
struct AddResult32 {
    sum: u32,
    carry: u32,
}

fn add_with_carry32(a: u32, b: u32, carry_in: u32) -> AddResult32 {
    let (s1, c1) = a.overflowing_add(b);
    let (s2, c2) = s1.overflowing_add(carry_in);
    AddResult32 {
        sum: s2,
        carry: (c1 as u32) | (c2 as u32),
    }
}

fn mul32_emulated(a: u32, b: u32) -> (u32, u32) {
    let a0 = a & 0xFFFF;
    let a1 = a >> 16;
    let b0 = b & 0xFFFF;
    let b1 = b >> 16;

    let c0 = a0.wrapping_mul(b0);

    let c1a = a0.wrapping_mul(b1);
    let c1b = a1.wrapping_mul(b0);
    let (c1_low, carry1) = c1a.overflowing_add(c1b);
    let c1_high_extra = if carry1 { 1u32 } else { 0u32 };

    let c2 = a1.wrapping_mul(b1);

    let m1_low16 = c1_low & 0xFFFF;
    let m1_high_total = (c1_low >> 16) + (c1_high_extra << 16);

    let low_part = m1_low16 << 16;
    let (low_tmp, carry_low0) = c0.overflowing_add(low_part);
    let carry0 = if carry_low0 { 1u32 } else { 0u32 };

    let lo = low_tmp;
    let hi = c2
        .wrapping_add(m1_high_total)
        .wrapping_add(carry0);

    (lo, hi)
}

// Compare a >= b (little-endian limbs).
fn geq_mod(limbs: &[u32; N_LIMBS], modulus: &[u32; N_LIMBS]) -> bool {
    for i in (0..N_LIMBS).rev() {
        let a = limbs[i];
        let m = modulus[i];
        if a > m {
            return true;
        }
        if a < m {
            return false;
        }
    }
    true
}

fn sub_mod_in_place(a: &mut [u32; N_LIMBS], b: &[u32; N_LIMBS]) {
    let mut borrow: u32 = 0;
    for i in 0..N_LIMBS {
        let ai = a[i];
        let bi = b[i];
        let sub = bi.wrapping_add(borrow);
        let diff = ai.wrapping_sub(sub);
        borrow = if ai < sub { 1 } else { 0 };
        a[i] = diff;
    }
}

fn conditional_sub_modulus(a: &mut [u32; N_LIMBS], modulus: &[u32; N_LIMBS]) {
    if geq_mod(a, modulus) {
        sub_mod_in_place(a, modulus);
    }
}

// BN254 scalar field modulus in base-2^32, little-endian, taken from bn254_sum.wgsl
const FR_MODULUS_32: [u32; N_LIMBS] = [
    4026531841u32,
    1138881939u32,
    2042196113u32,
    674490440u32,
    2172737629u32,
    3092268470u32,
    3778125865u32,
    811880050u32,
];

// -p^{-1} mod 2^32, precomputed in Rust once and hard-coded here.
// We'll compute this at runtime below, once, and print it for copying into WGSL.
fn compute_inv32() -> u32 {
    // Classic exponentiation: inv = -(p0)^{2^31 - 1} mod 2^32,
    // where p0 is p mod 2^32.
    let p0 = FR_MODULUS_32[0];
    let mut inv = 1u32;
    for _ in 0..31 {
        inv = inv.wrapping_mul(inv);
        inv = inv.wrapping_mul(p0);
    }
    inv.wrapping_neg()
}

// Montgomery multiplication in base 2^32, N_LIMBS = 8, using classic CIOS.
fn mont_mul(a: &Fr32, b: &Fr32, inv32: u32) -> Fr32 {
    let mut t = [0u32; N_LIMBS + 1]; // t[0..8], 9 limbs

    for i in 0..N_LIMBS {
        // t += a * b[i]
        let bi = b.limbs[i];
        let mut carry: u32 = 0;
        for j in 0..N_LIMBS {
            let (prod_lo, prod_hi) = mul32_emulated(a.limbs[j], bi);

            // t_j + prod_lo + carry
            let tmp1 = add_with_carry32(t[j], prod_lo, 0);
            let tmp2 = add_with_carry32(tmp1.sum, carry, 0);
            let c0 = tmp1.carry + tmp2.carry; // 0,1,2

            t[j] = tmp2.sum;
            carry = prod_hi
                .wrapping_add(c0);
        }
        t[N_LIMBS] = t[N_LIMBS].wrapping_add(carry);

        // m_i = (t[0] * inv32) mod 2^32
        let m_i = t[0].wrapping_mul(inv32);

        // t += m_i * p
        let mut carry2: u32 = 0;
        for j in 0..N_LIMBS {
            let (prod_lo, prod_hi) = mul32_emulated(m_i, FR_MODULUS_32[j]);

            let tmp1 = add_with_carry32(t[j], prod_lo, 0);
            let tmp2 = add_with_carry32(tmp1.sum, carry2, 0);
            let c0 = tmp1.carry + tmp2.carry;

            t[j] = tmp2.sum;
            carry2 = prod_hi
                .wrapping_add(c0);
        }
        t[N_LIMBS] = t[N_LIMBS].wrapping_add(carry2);

        // Shift t right by one 32-bit limb (divide by base).
        for k in 0..N_LIMBS {
            t[k] = t[k + 1];
        }
        t[N_LIMBS] = 0;
    }

    let mut out_limbs = [0u32; N_LIMBS];
    out_limbs.copy_from_slice(&t[..N_LIMBS]);
    conditional_sub_modulus(&mut out_limbs, &FR_MODULUS_32);
    Fr32 { limbs: out_limbs }
}

fn main() {
    let inv32 = compute_inv32();
    println!("Computed INV32 = {:#010x}", inv32);

    // Sanity-check INV32 against 64-bit arithmetic.
    let p0 = FR_MODULUS_32[0] as u64;
    let inv64 = inv32 as u64;
    let check = (p0 * inv64) & 0xFFFF_FFFF;
    println!("p0 * INV32 mod 2^32 = {:#010x}", check);

    let mut rng = rand::thread_rng();
    for _ in 0..10_000 {
        let a = Fr::rand(&mut rng);
        let b = Fr::rand(&mut rng);

        let a32 = fr_to_fr32_mont(a);
        let b32 = fr_to_fr32_mont(b);
        let c32 = mont_mul(&a32, &b32, inv32);
        let c = a * b;
        let c_from32 = fr32_mont_to_fr(&c32);

        if c != c_from32 {
            eprintln!("Mismatch:\na = {:?}\nb = {:?}\nCPU c = {:?}\nGPU-style c32 = {:?}\nback = {:?}",
                      a, b, c, c32.limbs, c_from32);
            std::process::exit(1);
        }
    }

    println!("BN254 Montgomery 8x32-bit mul matches ark_bn254::Fr for 10k random tests");
}


