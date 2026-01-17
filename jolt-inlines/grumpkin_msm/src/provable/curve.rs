//! Grumpkin curve operations on stable limb layouts.
//!
//! The Grumpkin curve is a short Weierstrass curve with `a = 0` and `b = -17`.
//! We implement Jacobian doubling, mixed addition (Jacobian + affine), and Jacobian addition.

use crate::types::{AffinePoint, JacobianPoint};

use super::fq::FqMont;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct AffineMont {
    pub x: FqMont,
    pub y: FqMont,
    pub infinity: bool,
}

impl AffineMont {
    #[inline(always)]
    pub fn from_affine_point(p: &AffinePoint) -> Self {
        if p.is_infinity() {
            return Self {
                x: FqMont::ZERO,
                y: FqMont::ZERO,
                infinity: true,
            };
        }
        Self {
            x: FqMont::from_canonical(p.x),
            y: FqMont::from_canonical(p.y),
            infinity: false,
        }
    }
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct JacobianMont {
    pub x: FqMont,
    pub y: FqMont,
    pub z: FqMont,
}

impl JacobianMont {
    #[inline(always)]
    pub const fn infinity() -> Self {
        Self {
            x: FqMont::ZERO,
            y: FqMont::ZERO,
            z: FqMont::ZERO,
        }
    }

    #[inline(always)]
    pub fn is_infinity(&self) -> bool {
        self.z.is_zero()
    }

    #[inline(always)]
    pub fn from_affine_mont(p: &AffineMont) -> Self {
        if p.infinity {
            return Self::infinity();
        }
        Self {
            x: p.x,
            y: p.y,
            z: FqMont::ONE,
        }
    }

    #[inline(always)]
    pub fn to_canonical(self) -> JacobianPoint {
        if self.is_infinity() {
            return JacobianPoint::infinity();
        }
        JacobianPoint {
            x: self.x.to_canonical(),
            y: self.y.to_canonical(),
            z: self.z.to_canonical(),
        }
    }
}

/// Jacobian point doubling for `a = 0` (EFD "dbl-2009-l" style).
#[inline(always)]
pub fn double_jac(p: JacobianMont) -> JacobianMont {
    if p.is_infinity() {
        return p;
    }

    // A = X1^2
    let a = p.x.square();
    // B = Y1^2
    let b = p.y.square();
    // C = B^2
    let c = b.square();

    // D = 2 * ((X1 + B)^2 - A - C)
    let x1_plus_b = p.x.add(b);
    let d = x1_plus_b.square().sub(a).sub(c).double();

    // E = 3*A
    let e = a.add(a).add(a);
    // F = E^2
    let f = e.square();

    // X3 = F - 2*D
    let x3 = f.sub(d.double());
    // Y3 = E*(D - X3) - 8*C
    let y3 = e.mul(d.sub(x3)).sub(c.double().double().double());
    // Z3 = 2*Y1*Z1
    let z3 = p.y.mul(p.z).double();

    JacobianMont {
        x: x3,
        y: y3,
        z: z3,
    }
}

/// Mixed addition: `out = jac + aff` (Jacobian + affine).
#[inline(always)]
pub fn add_mixed(jac: JacobianMont, aff: &AffineMont) -> JacobianMont {
    if jac.is_infinity() {
        return JacobianMont::from_affine_mont(aff);
    }
    if aff.infinity {
        return jac;
    }

    // Z1Z1 = Z1^2
    let z1z1 = jac.z.square();
    // U2 = x2 * Z1Z1
    let u2 = aff.x.mul(z1z1);
    // S2 = y2 * Z1 * Z1Z1
    let s2 = aff.y.mul(jac.z.mul(z1z1));

    // H = U2 - X1
    let h = u2.sub(jac.x);
    // r = 2*(S2 - Y1)
    let r = s2.sub(jac.y).double();

    if h.is_zero() {
        // If H == 0:
        // - if r == 0, points are equal -> doubling
        // - else, P + (-P) = O
        if r.is_zero() {
            return double_jac(jac);
        }
        return JacobianMont::infinity();
    }

    // HH = H^2
    let hh = h.square();
    // I = 4*HH
    let i = hh.double().double();
    // J = H*I
    let j = h.mul(i);
    // V = X1*I
    let v = jac.x.mul(i);

    // X3 = r^2 - J - 2*V
    let x3 = r.square().sub(j).sub(v.double());
    // Y3 = r*(V - X3) - 2*Y1*J
    let y3 = r.mul(v.sub(x3)).sub(jac.y.mul(j).double());
    // Z3 = (Z1 + H)^2 - Z1Z1 - HH
    let z3 = jac.z.add(h).square().sub(z1z1).sub(hh);

    JacobianMont {
        x: x3,
        y: y3,
        z: z3,
    }
}

/// Jacobian + Jacobian addition.
#[inline(always)]
pub fn add_jac(p: JacobianMont, q: JacobianMont) -> JacobianMont {
    if p.is_infinity() {
        return q;
    }
    if q.is_infinity() {
        return p;
    }

    // Z1Z1 = Z1^2, Z2Z2 = Z2^2
    let z1z1 = p.z.square();
    let z2z2 = q.z.square();

    // U1 = X1*Z2Z2, U2 = X2*Z1Z1
    let u1 = p.x.mul(z2z2);
    let u2 = q.x.mul(z1z1);

    // S1 = Y1*Z2*Z2Z2, S2 = Y2*Z1*Z1Z1
    let s1 = p.y.mul(q.z.mul(z2z2));
    let s2 = q.y.mul(p.z.mul(z1z1));

    // H = U2 - U1
    let h = u2.sub(u1);
    // r = 2*(S2 - S1)
    let r = s2.sub(s1).double();

    if h.is_zero() {
        if r.is_zero() {
            return double_jac(p);
        }
        return JacobianMont::infinity();
    }

    // I = (2H)^2
    let i = h.double().square();
    // J = H*I
    let j = h.mul(i);
    // V = U1*I
    let v = u1.mul(i);

    // X3 = r^2 - J - 2V
    let x3 = r.square().sub(j).sub(v.double());
    // Y3 = r*(V - X3) - 2*S1*J
    let y3 = r.mul(v.sub(x3)).sub(s1.mul(j).double());
    // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H
    let z3 = p.z.add(q.z).square().sub(z1z1).sub(z2z2).mul(h);

    JacobianMont {
        x: x3,
        y: y3,
        z: z3,
    }
}
