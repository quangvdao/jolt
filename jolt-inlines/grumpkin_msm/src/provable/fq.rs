//! Base-field arithmetic for Grumpkin (Fq) on 4×64-bit limbs.
//!
//! Grumpkin's base field is `ark_bn254::Fr` (see `ark_grumpkin::Fq`), i.e. the BN254 scalar field.
//! We implement arithmetic in **Montgomery form** to make multiplication cheap.

use crate::types::FqLimbs;

/// BN254 Fr modulus (little-endian u64 limbs).
///
/// Decimal:
/// `21888242871839275222246405745257275088548364400416034343698204186575808495617`
const MODULUS: [u64; 4] = [
    0x43e1_f593_f000_0001,
    0x2833_e848_79b9_7091,
    0xb850_45b6_8181_585d,
    0x3064_4e72_e131_a029,
];

/// `-MODULUS^{-1} mod 2^64` (Montgomery reduction constant).
const INV: u64 = 0xc2e1_f593_efff_ffff;

/// `R^2 mod MODULUS`, where `R = 2^256`.
const R2: [u64; 4] = [
    0x1bb8_e645_ae21_6da7,
    0x53fe_3ab1_e35c_59e3,
    0x8c49_833d_53bb_8085,
    0x0216_d0b1_7f4e_44a5,
];

/// `R mod MODULUS` (Montgomery representation of 1).
const ONE_MONT: [u64; 4] = [
    0xac96_341c_4fff_fffb,
    0x36fc_7695_9f60_cd29,
    0x666e_a36f_7879_462e,
    0x0e0a_77c1_9a07_df2f,
];

#[inline(always)]
fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let sum = (a as u128) + (b as u128) + (carry as u128);
    (sum as u64, (sum >> 64) as u64)
}

#[inline(always)]
fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let (res1, borrow1) = a.overflowing_sub(b);
    let (res2, borrow2) = res1.overflowing_sub(borrow);
    (res2, (borrow1 as u64) | (borrow2 as u64))
}

#[inline(always)]
fn geq_256(a: &[u64; 4], b: &[u64; 4]) -> bool {
    for i in (0..4).rev() {
        if a[i] > b[i] {
            return true;
        }
        if a[i] < b[i] {
            return false;
        }
    }
    true
}

#[inline(always)]
fn add_256(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], u64) {
    let mut out = [0u64; 4];
    let mut carry = 0u64;
    for i in 0..4 {
        (out[i], carry) = adc(a[i], b[i], carry);
    }
    (out, carry)
}

#[inline(always)]
fn sub_256(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], u64) {
    let mut out = [0u64; 4];
    let mut borrow = 0u64;
    for i in 0..4 {
        (out[i], borrow) = sbb(a[i], b[i], borrow);
    }
    (out, borrow)
}

#[inline(always)]
fn montgomery_mul(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    // Schoolbook multiply: 256×256 → 512 (8 limbs).
    let mut t = [0u64; 8];
    for i in 0..4 {
        let mut carry = 0u128;
        for j in 0..4 {
            let idx = i + j;
            let acc = (a[i] as u128) * (b[j] as u128) + (t[idx] as u128) + carry;
            t[idx] = acc as u64;
            carry = acc >> 64;
        }
        // Add carry into upper limb and propagate (bounded to 8 limbs).
        let mut carry64 = carry as u64;
        let mut k = i + 4;
        (t[k], carry64) = adc(t[k], carry64, 0);
        k += 1;
        for kk in k..8 {
            if carry64 == 0 {
                break;
            }
            (t[kk], carry64) = adc(t[kk], 0, carry64);
        }
        debug_assert_eq!(carry64, 0, "256x256 multiplication overflowed 512 bits");
    }

    // Montgomery reduction (4 iterations).
    for i in 0..4 {
        let m = t[i].wrapping_mul(INV);
        let mut carry = 0u128;
        for j in 0..4 {
            let idx = i + j;
            let acc = (m as u128) * (MODULUS[j] as u128) + (t[idx] as u128) + carry;
            t[idx] = acc as u64;
            carry = acc >> 64;
        }

        // Add carry into t[i+4] and propagate.
        let mut carry64 = carry as u64;
        let mut k = i + 4;
        (t[k], carry64) = adc(t[k], carry64, 0);
        k += 1;
        for kk in k..8 {
            if carry64 == 0 {
                break;
            }
            (t[kk], carry64) = adc(t[kk], 0, carry64);
        }
        debug_assert_eq!(carry64, 0, "Montgomery reduction carry overflow");
    }

    let mut out = [t[4], t[5], t[6], t[7]];
    if geq_256(&out, &MODULUS) {
        out = sub_256(&out, &MODULUS).0;
    }
    out
}

/// Field element in Montgomery form.
#[repr(transparent)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct FqMont(pub [u64; 4]);

impl FqMont {
    pub const ZERO: Self = Self([0u64; 4]);
    pub const ONE: Self = Self(ONE_MONT);

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.0 == [0u64; 4]
    }

    /// Convert a canonical (non-Montgomery) limb encoding into Montgomery form.
    #[inline(always)]
    pub fn from_canonical(c: FqLimbs) -> Self {
        Self(montgomery_mul(&c.0, &R2))
    }

    /// Convert a Montgomery-form element back into canonical limbs.
    #[inline(always)]
    pub fn to_canonical(self) -> FqLimbs {
        FqLimbs(montgomery_mul(&self.0, &[1, 0, 0, 0]))
    }

    #[inline(always)]
    pub fn add(self, rhs: Self) -> Self {
        let (sum, carry) = add_256(&self.0, &rhs.0);
        let mut out = sum;
        if carry != 0 || geq_256(&out, &MODULUS) {
            out = sub_256(&out, &MODULUS).0;
        }
        Self(out)
    }

    #[inline(always)]
    pub fn sub(self, rhs: Self) -> Self {
        let (diff, borrow) = sub_256(&self.0, &rhs.0);
        let mut out = diff;
        if borrow != 0 {
            out = add_256(&out, &MODULUS).0;
        }
        Self(out)
    }

    #[inline(always)]
    pub fn double(self) -> Self {
        self.add(self)
    }

    #[inline(always)]
    pub fn mul(self, rhs: Self) -> Self {
        Self(montgomery_mul(&self.0, &rhs.0))
    }

    #[inline(always)]
    pub fn square(self) -> Self {
        self.mul(self)
    }
}
