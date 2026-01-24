//! G2 addition witness generation for Dory recursion
//!
//! Same as G1 but with Fq2 coordinates split into (c0, c1) components
//! since the recursion SNARK runs over the base field Fq.

use ark_bn254::{Fq, Fq2, G2Affine};
use ark_ec::AffineRepr;
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::backends::arkworks::ArkG2;
use dory::recursion::WitnessResult;

/// G2 addition witness for Dory recursion.
///
/// Same as G1 but with Fq2 coordinates split into (c0, c1) components.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G2AdditionSteps {
    // Input P (Fq2 coordinates)
    pub x_p_c0: Fq,
    pub x_p_c1: Fq,
    pub y_p_c0: Fq,
    pub y_p_c1: Fq,
    pub ind_p: Fq,

    // Input Q
    pub x_q_c0: Fq,
    pub x_q_c1: Fq,
    pub y_q_c0: Fq,
    pub y_q_c1: Fq,
    pub ind_q: Fq,

    // Output R = P + Q
    pub x_r_c0: Fq,
    pub x_r_c1: Fq,
    pub y_r_c0: Fq,
    pub y_r_c1: Fq,
    pub ind_r: Fq,

    // Auxiliary witness values (Fq2)
    pub lambda_c0: Fq,
    pub lambda_c1: Fq,
    pub inv_delta_x_c0: Fq,
    pub inv_delta_x_c1: Fq,
    pub is_double: Fq,
    pub is_inverse: Fq,

    // For WitnessResult trait
    ark_result: ArkG2,
}

impl WitnessResult<ArkG2> for G2AdditionSteps {
    fn result(&self) -> Option<&ArkG2> {
        Some(&self.ark_result)
    }
}

impl G2AdditionSteps {
    /// Generate witness for P + Q = R in G2
    pub fn new(a: &ArkG2, b: &ArkG2, result: &ArkG2) -> Self {
        let p: G2Affine = a.0.into();
        let q: G2Affine = b.0.into();
        let r: G2Affine = result.0.into();

        let zero = Fq::from(0u64);
        let one = Fq::from(1u64);
        let fq2_zero = Fq2::from(0u64);

        // Extract Fq2 coordinates split into (c0, c1)
        let (x_p, y_p, ind_p) = if p.is_zero() {
            (fq2_zero, fq2_zero, one)
        } else {
            (p.x, p.y, zero)
        };

        let (x_q, y_q, ind_q) = if q.is_zero() {
            (fq2_zero, fq2_zero, one)
        } else {
            (q.x, q.y, zero)
        };

        let (x_r, y_r, ind_r) = if r.is_zero() {
            (fq2_zero, fq2_zero, one)
        } else {
            (r.x, r.y, zero)
        };

        // Compute auxiliary witness values in Fq2
        let (lambda, inv_delta_x, is_double, is_inverse) = if ind_p == one || ind_q == one {
            (fq2_zero, fq2_zero, zero, zero)
        } else {
            let dx = x_q - x_p;
            let dy = y_q - y_p;

            if dx == fq2_zero {
                if dy == fq2_zero {
                    // Doubling case
                    let two = Fq2::from(2u64);
                    let three = Fq2::from(3u64);
                    let numerator = three * x_p * x_p;
                    let denominator = two * y_p;
                    let lam = if denominator == fq2_zero {
                        fq2_zero
                    } else {
                        numerator * denominator.inverse().unwrap()
                    };
                    (lam, fq2_zero, one, zero)
                } else {
                    // Inverse case
                    (fq2_zero, fq2_zero, zero, one)
                }
            } else {
                // General add
                let inv_dx = dx.inverse().unwrap();
                let lam = dy * inv_dx;
                (lam, inv_dx, zero, zero)
            }
        };

        Self {
            x_p_c0: x_p.c0,
            x_p_c1: x_p.c1,
            y_p_c0: y_p.c0,
            y_p_c1: y_p.c1,
            ind_p,
            x_q_c0: x_q.c0,
            x_q_c1: x_q.c1,
            y_q_c0: y_q.c0,
            y_q_c1: y_q.c1,
            ind_q,
            x_r_c0: x_r.c0,
            x_r_c1: x_r.c1,
            y_r_c0: y_r.c0,
            y_r_c1: y_r.c1,
            ind_r,
            lambda_c0: lambda.c0,
            lambda_c1: lambda.c1,
            inv_delta_x_c0: inv_delta_x.c0,
            inv_delta_x_c1: inv_delta_x.c1,
            is_double,
            is_inverse,
            ark_result: *result,
        }
    }
}
