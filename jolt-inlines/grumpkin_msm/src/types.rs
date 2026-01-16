//! Common utilities and stable memory layouts for Grumpkin MSM.
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

// --------------------------
// Host-side conversions
// --------------------------

#[cfg(feature = "host")]
mod host_conv {
    use super::*;
    use ark_ec::AffineRepr;
    use ark_ff::{BigInt, PrimeField};

    impl FqLimbs {
        #[inline]
        pub fn from_ark_fq(f: &ark_grumpkin::Fq) -> Self {
            Self(f.into_bigint().0)
        }

        #[inline]
        pub fn try_into_ark_fq(&self) -> Option<ark_grumpkin::Fq> {
            let bigint = BigInt::new(self.0);
            <ark_grumpkin::Fq as PrimeField>::from_bigint(bigint)
        }
    }

    impl FrLimbs {
        #[inline]
        pub fn from_ark_fr(f: &ark_grumpkin::Fr) -> Self {
            Self(f.into_bigint().0)
        }

        #[inline]
        pub fn try_into_ark_fr(&self) -> Option<ark_grumpkin::Fr> {
            let bigint = BigInt::new(self.0);
            <ark_grumpkin::Fr as PrimeField>::from_bigint(bigint)
        }
    }

    impl AffinePoint {
        #[inline]
        pub fn from_ark_affine(p: &ark_grumpkin::Affine) -> Self {
            if p.is_zero() {
                return Self::infinity();
            }
            Self {
                x: FqLimbs::from_ark_fq(&p.x),
                y: FqLimbs::from_ark_fq(&p.y),
                infinity: 0,
            }
        }

        #[inline]
        pub fn try_into_ark_affine(&self) -> Option<ark_grumpkin::Affine> {
            if self.is_infinity() {
                return Some(ark_grumpkin::Affine::identity());
            }
            let x = self.x.try_into_ark_fq()?;
            let y = self.y.try_into_ark_fq()?;
            Some(ark_grumpkin::Affine::new_unchecked(x, y))
        }
    }

    impl JacobianPoint {
        #[inline]
        pub fn from_ark_projective(p: &ark_grumpkin::Projective) -> Self {
            Self {
                x: FqLimbs::from_ark_fq(&p.x),
                y: FqLimbs::from_ark_fq(&p.y),
                z: FqLimbs::from_ark_fq(&p.z),
            }
        }

        #[inline]
        pub fn try_into_ark_projective(&self) -> Option<ark_grumpkin::Projective> {
            let x = self.x.try_into_ark_fq()?;
            let y = self.y.try_into_ark_fq()?;
            let z = self.z.try_into_ark_fq()?;
            Some(ark_grumpkin::Projective { x, y, z })
        }
    }

    // --------------------------
    // Tests (host-only)
    // --------------------------

    #[cfg(test)]
    mod tests {
        use super::*;
        use ark_ff::UniformRand;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        #[test]
        fn fq_roundtrip_limbs() {
            let mut rng = StdRng::seed_from_u64(0xC0FFEE);
            for _ in 0..100 {
                let fq = ark_grumpkin::Fq::rand(&mut rng);
                let limbs = FqLimbs::from_ark_fq(&fq);
                let back = limbs.try_into_ark_fq().expect("canonical limbs");
                assert_eq!(fq, back);
            }
        }

        #[test]
        fn fr_roundtrip_limbs() {
            let mut rng = StdRng::seed_from_u64(0xBADC0DE);
            for _ in 0..100 {
                let fr = ark_grumpkin::Fr::rand(&mut rng);
                let limbs = FrLimbs::from_ark_fr(&fr);
                let back = limbs.try_into_ark_fr().expect("canonical limbs");
                assert_eq!(fr, back);
            }
        }

        #[test]
        fn affine_roundtrip_limbs_including_infinity() {
            let mut rng = StdRng::seed_from_u64(12345);

            // Infinity case.
            let inf = ark_grumpkin::Affine::zero();
            let inf_limbs = AffinePoint::from_ark_affine(&inf);
            assert!(inf_limbs.is_infinity());
            assert_eq!(
                inf,
                inf_limbs
                    .try_into_ark_affine()
                    .expect("infinity conversion should succeed")
            );

            for _ in 0..50 {
                let p = ark_ec::CurveGroup::into_affine(ark_grumpkin::Projective::rand(&mut rng));
                let limbs = AffinePoint::from_ark_affine(&p);
                let back = limbs.try_into_ark_affine().expect("canonical point limbs");
                assert_eq!(p, back);
            }
        }

        #[test]
        fn fr_window_extraction_smoke() {
            // Bit pattern: limb0 = 0b...1001 (LSB-first), limb1 all zeros.
            let fr = FrLimbs([0b1001, 0, 0, 0]);
            assert_eq!(fr.get_bit(0), true);
            assert_eq!(fr.get_bit(1), false);
            assert_eq!(fr.get_bit(3), true);

            // windows of size 2: bits [0..2) = 01, [2..4) = 10.
            assert_eq!(fr.window(0, 2), 1);
            assert_eq!(fr.window(2, 2), 2);
        }
    }
}

