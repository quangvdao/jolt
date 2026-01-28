//! G2 curve operations sumchecks
//!
//! This module contains sumcheck protocols for G2 group operations:
//! - Addition: Proves G2 point addition constraints
//! - Scalar multiplication: Proves G2 scalar multiplication constraints
//!
//! ## Fq2 Arithmetic over Fq
//!
//! G2 points live over the quadratic extension field Fq2 = Fq[u]/(u^2 + 1).
//! Since our recursion SNARK runs over the base field Fq, we split each Fq2
//! coordinate into its (c0, c1) components and enforce constraints component-wise.
//!
//! The [`Fq2Components`] struct provides component-wise Fq2 arithmetic with
//! zero-overhead methods for when only one component is needed.

pub mod fused_addition;
pub mod fused_scalar_multiplication;
pub mod fused_wiring;
pub mod indexing;
pub mod types;

pub use fused_addition::{FusedG2AddParams, FusedG2AddProver, FusedG2AddVerifier};
pub use fused_scalar_multiplication::{
    FusedG2ScalarMulProver, FusedG2ScalarMulVerifier, FusedShiftG2ScalarMulProver,
    FusedShiftG2ScalarMulVerifier,
};
pub use types::G2ScalarMulPublicInputs;

use crate::field::JoltField;

/// Represents an Fq2 element split into (c0, c1) components over a base field F.
///
/// Fq2 = Fq[u]/(u^2 + 1), so (c0, c1) represents c0 + c1*u where u^2 = -1.
///
/// This struct provides arithmetic operations that work component-wise over F,
/// with specialized methods for extracting just one component when needed
/// (avoiding unnecessary computation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fq2Components<F> {
    pub c0: F,
    pub c1: F,
}

impl<F: JoltField> Fq2Components<F> {
    /// Create a new Fq2Components from c0 and c1.
    #[inline(always)]
    pub fn new(c0: F, c1: F) -> Self {
        Self { c0, c1 }
    }

    /// Create zero element.
    #[inline(always)]
    pub fn zero() -> Self {
        Self {
            c0: F::zero(),
            c1: F::zero(),
        }
    }

    /// Create one element (1 + 0*u).
    #[inline(always)]
    pub fn one() -> Self {
        Self {
            c0: F::one(),
            c1: F::zero(),
        }
    }

    /// Create from a base field element (embedded as c0, with c1 = 0).
    #[inline(always)]
    pub fn from_base(val: F) -> Self {
        Self {
            c0: val,
            c1: F::zero(),
        }
    }

    // =========================================================================
    // Full arithmetic operations (return Fq2Components)
    // =========================================================================

    /// Add two Fq2 elements: (a0,a1) + (b0,b1) = (a0+b0, a1+b1)
    #[inline(always)]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            c0: self.c0 + other.c0,
            c1: self.c1 + other.c1,
        }
    }

    /// Subtract two Fq2 elements: (a0,a1) - (b0,b1) = (a0-b0, a1-b1)
    #[inline(always)]
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            c0: self.c0 - other.c0,
            c1: self.c1 - other.c1,
        }
    }

    /// Negate an Fq2 element: -(a0,a1) = (-a0, -a1)
    #[inline(always)]
    pub fn neg(&self) -> Self {
        Self {
            c0: -self.c0,
            c1: -self.c1,
        }
    }

    /// Multiply two Fq2 elements: (a0,a1) * (b0,b1)
    /// = (a0*b0 - a1*b1, a0*b1 + a1*b0)
    #[inline(always)]
    pub fn mul(&self, other: &Self) -> Self {
        Self {
            c0: self.c0 * other.c0 - self.c1 * other.c1,
            c1: self.c0 * other.c1 + self.c1 * other.c0,
        }
    }

    /// Square an Fq2 element: (a0,a1)^2 = (a0^2 - a1^2, 2*a0*a1)
    #[inline(always)]
    pub fn square(&self) -> Self {
        Self {
            c0: self.c0 * self.c0 - self.c1 * self.c1,
            c1: F::from_u64(2) * self.c0 * self.c1,
        }
    }

    /// Multiply by a base field scalar: (a0,a1) * s = (a0*s, a1*s)
    #[inline(always)]
    pub fn scale(&self, scalar: F) -> Self {
        Self {
            c0: self.c0 * scalar,
            c1: self.c1 * scalar,
        }
    }

    // =========================================================================
    // Single-component operations (return F, avoid computing unused component)
    // =========================================================================

    /// Compute only c0 of multiplication: a0*b0 - a1*b1
    #[inline(always)]
    pub fn mul_c0(&self, other: &Self) -> F {
        self.c0 * other.c0 - self.c1 * other.c1
    }

    /// Compute only c1 of multiplication: a0*b1 + a1*b0
    #[inline(always)]
    pub fn mul_c1(&self, other: &Self) -> F {
        self.c0 * other.c1 + self.c1 * other.c0
    }

    /// Compute only c0 of squaring: a0^2 - a1^2
    #[inline(always)]
    pub fn sq_c0(&self) -> F {
        self.c0 * self.c0 - self.c1 * self.c1
    }

    /// Compute only c1 of squaring: 2*a0*a1
    #[inline(always)]
    pub fn sq_c1(&self) -> F {
        F::from_u64(2) * self.c0 * self.c1
    }
}

/// Compute the c0 component of Fq2 multiplication: (a0,a1) * (b0,b1).
/// Returns: a0*b0 - a1*b1
#[inline(always)]
pub fn fq2_mul_c0<F: JoltField>(a0: F, a1: F, b0: F, b1: F) -> F {
    a0 * b0 - a1 * b1
}

/// Compute the c1 component of Fq2 multiplication: (a0,a1) * (b0,b1).
/// Returns: a0*b1 + a1*b0
#[inline(always)]
pub fn fq2_mul_c1<F: JoltField>(a0: F, a1: F, b0: F, b1: F) -> F {
    a0 * b1 + a1 * b0
}

/// Compute the c0 component of Fq2 squaring: (a0,a1)^2.
/// Returns: a0^2 - a1^2
#[inline(always)]
pub fn fq2_sq_c0<F: JoltField>(a0: F, a1: F) -> F {
    a0 * a0 - a1 * a1
}

/// Compute the c1 component of Fq2 squaring: (a0,a1)^2.
/// Returns: 2*a0*a1
#[inline(always)]
pub fn fq2_sq_c1<F: JoltField>(a0: F, a1: F) -> F {
    F::from_u64(2) * a0 * a1
}
