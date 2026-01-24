//! G1 addition witness generation for Dory recursion
//!
//! Computes auxiliary witness values (lambda, inv_delta_x, branch flags) for
//! proving G1 point addition constraints.

use ark_bn254::{Fq, G1Affine};
use ark_ec::AffineRepr;
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use dory::backends::arkworks::ArkG1;
use dory::recursion::WitnessResult;

/// G1 addition witness for Dory recursion.
///
/// For a single point addition P + Q = R, we store:
/// - The coordinates and infinity indicators for P, Q, R
/// - Auxiliary values: lambda (slope), inv_delta_x, is_double, is_inverse
///
/// Since G1Add is a single operation (not a trace), the MLE is constant:
/// all 2^11 entries have the same value.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G1AdditionSteps {
    // Input P
    pub x_p: Fq,
    pub y_p: Fq,
    pub ind_p: Fq, // 1 if P = O (infinity), 0 otherwise

    // Input Q
    pub x_q: Fq,
    pub y_q: Fq,
    pub ind_q: Fq,

    // Output R = P + Q
    pub x_r: Fq,
    pub y_r: Fq,
    pub ind_r: Fq,

    // Auxiliary witness values
    pub lambda: Fq,      // slope
    pub inv_delta_x: Fq, // 1/(x_q - x_p) in add case
    pub is_double: Fq,   // 1 if P == Q
    pub is_inverse: Fq,  // 1 if P == -Q (result is O)

    // For WitnessResult trait
    ark_result: ArkG1,
}

impl WitnessResult<ArkG1> for G1AdditionSteps {
    fn result(&self) -> Option<&ArkG1> {
        Some(&self.ark_result)
    }
}

impl G1AdditionSteps {
    /// Generate witness for P + Q = R
    pub fn new(a: &ArkG1, b: &ArkG1, result: &ArkG1) -> Self {
        let p: G1Affine = a.0.into();
        let q: G1Affine = b.0.into();
        let r: G1Affine = result.0.into();

        let zero = Fq::from(0u64);
        let one = Fq::from(1u64);

        // Extract coordinates (0 for infinity points)
        let (x_p, y_p, ind_p) = if p.is_zero() {
            (zero, zero, one)
        } else {
            (p.x, p.y, zero)
        };

        let (x_q, y_q, ind_q) = if q.is_zero() {
            (zero, zero, one)
        } else {
            (q.x, q.y, zero)
        };

        let (x_r, y_r, ind_r) = if r.is_zero() {
            (zero, zero, one)
        } else {
            (r.x, r.y, zero)
        };

        // Compute auxiliary witness values
        let (lambda, inv_delta_x, is_double, is_inverse) = if ind_p == one || ind_q == one {
            // At least one input is infinity - no slope needed
            (zero, zero, zero, zero)
        } else {
            let dx = x_q - x_p;
            let dy = y_q - y_p;

            if dx == zero {
                if dy == zero {
                    // P == Q: doubling case
                    // lambda = 3*x_p^2 / (2*y_p)
                    let two = Fq::from(2u64);
                    let three = Fq::from(3u64);
                    let numerator = three * x_p * x_p;
                    let denominator = two * y_p;
                    let lam = if denominator == zero {
                        zero // Edge case: y_p = 0 means P = -P, result is O
                    } else {
                        numerator * denominator.inverse().unwrap()
                    };
                    (lam, zero, one, zero)
                } else {
                    // P == -Q: inverse case, result is infinity
                    (zero, zero, zero, one)
                }
            } else {
                // General add case: lambda = dy/dx
                let inv_dx = dx.inverse().unwrap();
                let lam = dy * inv_dx;
                (lam, inv_dx, zero, zero)
            }
        };

        Self {
            x_p,
            y_p,
            ind_p,
            x_q,
            y_q,
            ind_q,
            x_r,
            y_r,
            ind_r,
            lambda,
            inv_delta_x,
            is_double,
            is_inverse,
            ark_result: *result,
        }
    }
}
