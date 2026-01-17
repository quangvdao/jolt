//! Common utilities and stable memory layouts for Grumpkin MSM / curve ops.
//!
//! The intent is to avoid relying on Rust/arkworks struct layout in guest memory by using
//! explicit `#[repr(C)]` encodings.

/// Number of 64-bit limbs in Grumpkin base/scalar field elements.
pub const LIMBS_64: usize = 4;

/// Base field limbs (Grumpkin Fq) in little-endian u64 limb order.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct FqLimbs(pub [u64; LIMBS_64]);

/// Scalar field limbs (Grumpkin Fr) in little-endian u64 limb order.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct FrLimbs(pub [u64; LIMBS_64]);

impl FrLimbs {
    /// Returns the bit at `bit_idx` (LSB-first). Out-of-range bits are `false`.
    #[inline(always)]
    pub fn get_bit(&self, bit_idx: usize) -> bool {
        let limb = bit_idx / 64;
        if limb >= LIMBS_64 {
            return false;
        }
        let shift = bit_idx % 64;
        ((self.0[limb] >> shift) & 1) == 1
    }

    /// Extract a window of `window_size` bits starting at `start_bit` (LSB-first).
    ///
    /// - Returns a value in `[0, 2^window_size)`.
    /// - Out-of-range bits are treated as 0.
    #[inline(always)]
    pub fn window(&self, start_bit: usize, window_size: usize) -> u32 {
        debug_assert!(window_size > 0);
        debug_assert!(window_size <= 32);

        let limb = start_bit / 64;
        let shift = start_bit % 64;
        if limb >= LIMBS_64 {
            return 0;
        }

        let mut word = self.0[limb] >> shift;
        if shift != 0 && limb + 1 < LIMBS_64 {
            word |= self.0[limb + 1] << (64 - shift);
        }

        let mask = if window_size == 32 {
            u64::MAX
        } else {
            (1u64 << window_size) - 1
        };
        (word & mask) as u32
    }
}

/// Stable affine point encoding.
///
/// - `infinity != 0` encodes the point at infinity (x/y are ignored).
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct AffinePoint {
    pub x: FqLimbs,
    pub y: FqLimbs,
    pub infinity: u64,
}

impl AffinePoint {
    #[inline(always)]
    pub const fn infinity() -> Self {
        Self {
            x: FqLimbs([0; LIMBS_64]),
            y: FqLimbs([0; LIMBS_64]),
            infinity: 1,
        }
    }

    #[inline(always)]
    pub const fn is_infinity(&self) -> bool {
        self.infinity != 0
    }
}

/// Stable Jacobian point encoding.
///
/// - Infinity is represented as `z == 0`.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct JacobianPoint {
    pub x: FqLimbs,
    pub y: FqLimbs,
    pub z: FqLimbs,
}

impl JacobianPoint {
    #[inline(always)]
    pub const fn infinity() -> Self {
        Self {
            x: FqLimbs([0; LIMBS_64]),
            y: FqLimbs([0; LIMBS_64]),
            z: FqLimbs([0; LIMBS_64]),
        }
    }

    #[inline(always)]
    pub fn is_infinity(&self) -> bool {
        self.z.0 == [0; LIMBS_64]
    }
}
