//! G2 scalar multiplication witness generation for Dory recursion
//! Implements the double-and-add algorithm for elliptic curve scalar multiplication in G2.
//!
//! G2 points live over Fq2, so we split each Fq2 coordinate into its (c0, c1) components in Fq.

use ark_bn254::{Fq, Fq2, Fr, G2Affine, G2Projective};
use ark_ec::AffineRepr;
use ark_ff::{BigInteger, One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::ops::Add;

/// Build an 8-var MLE from a vector of per-step field elements.
/// Pads with zeros to reach the full size of the Boolean hypercube.
fn build_mle_from_steps(step_values: &[Fq], num_vars: usize) -> Vec<Fq> {
    let size = 1 << num_vars;
    let mut mle = vec![Fq::zero(); size];

    for (i, &value) in step_values.iter().enumerate() {
        if i < size {
            mle[i] = value;
        }
    }

    mle
}

#[inline]
fn split_fq2_steps(step_values: &[Fq2]) -> (Vec<Fq>, Vec<Fq>) {
    let mut c0 = Vec::with_capacity(step_values.len());
    let mut c1 = Vec::with_capacity(step_values.len());
    for v in step_values {
        c0.push(v.c0);
        c1.push(v.c1);
    }
    (c0, c1)
}

/// G2 scalar multiplication witness generation.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G2ScalarMultiplicationSteps {
    pub point_base: G2Affine, // Base point P
    pub scalar: Fr,           // Scalar k
    pub result: G2Affine,     // Result Q = [k]P

    // Witness MLEs (8 variables, 256 steps)
    // Fq2 coordinates are split into (c0, c1) over Fq.
    pub x_a_c0_mles: Vec<Vec<Fq>>,
    pub x_a_c1_mles: Vec<Vec<Fq>>,
    pub y_a_c0_mles: Vec<Vec<Fq>>,
    pub y_a_c1_mles: Vec<Vec<Fq>>,

    pub x_t_c0_mles: Vec<Vec<Fq>>,
    pub x_t_c1_mles: Vec<Vec<Fq>>,
    pub y_t_c0_mles: Vec<Vec<Fq>>,
    pub y_t_c1_mles: Vec<Vec<Fq>>,

    pub x_a_next_c0_mles: Vec<Vec<Fq>>,
    pub x_a_next_c1_mles: Vec<Vec<Fq>>,
    pub y_a_next_c0_mles: Vec<Vec<Fq>>,
    pub y_a_next_c1_mles: Vec<Vec<Fq>>,

    /// Indicator: 1 if T_i = O (point at infinity), 0 otherwise
    pub t_is_infinity_mles: Vec<Vec<Fq>>,
    /// Indicator: 1 if A_i = O (point at infinity), 0 otherwise
    pub a_is_infinity_mles: Vec<Vec<Fq>>,
    /// Scalar bit b_i ∈ {0,1} as Fq element (MSB first)
    pub bit_mles: Vec<Vec<Fq>>,

    pub bits: Vec<bool>, // Scalar bits (MSB first, always 256 bits)
}

impl G2ScalarMultiplicationSteps {
    /// Generate witness for [scalar]point using double-and-add algorithm (MSB-first, 256 steps).
    pub fn new(point: G2Affine, scalar: Fr) -> Self {
        // Get binary representation of scalar (little-endian)
        let scalar_bits = scalar.into_bigint().to_bits_le();

        // Always process exactly 256 bits (8-variate MLE, 2^8 = 256)
        let bits_msb: Vec<bool> = (0..256).rev().map(|i| scalar_bits[i]).collect();
        let num_vars = 8;

        let n = bits_msb.len();
        let mut x_a_values: Vec<Fq2> = Vec::with_capacity(n + 1); // A_0..A_n
        let mut y_a_values: Vec<Fq2> = Vec::with_capacity(n + 1);
        let mut x_t_values: Vec<Fq2> = Vec::with_capacity(n); // T_0..T_{n-1}
        let mut y_t_values: Vec<Fq2> = Vec::with_capacity(n);
        let mut t_is_infinity_values: Vec<Fq> = Vec::with_capacity(n);
        let mut a_is_infinity_values: Vec<Fq> = Vec::with_capacity(n);
        let mut bit_values: Vec<Fq> = Vec::with_capacity(n);

        // Initialize accumulator with point at infinity (identity)
        let mut accumulator = G2Projective::zero();

        // Store A_0 = O
        let a_0: G2Affine = accumulator.into();
        let (x_a_0, y_a_0) = if a_0.is_zero() {
            (Fq2::zero(), Fq2::zero())
        } else {
            (a_0.x, a_0.y)
        };
        x_a_values.push(x_a_0);
        y_a_values.push(y_a_0);

        // Double-and-add over MSB-first bits
        for &bit in bits_msb.iter() {
            // Record if A_i is infinity (before doubling)
            let a_affine: G2Affine = accumulator.into();
            let a_is_inf = if a_affine.is_zero() {
                Fq::one()
            } else {
                Fq::zero()
            };
            a_is_infinity_values.push(a_is_inf);

            // Record bit as field element
            bit_values.push(if bit { Fq::one() } else { Fq::zero() });

            // Double: T_i = [2]A_i
            let doubled = accumulator + accumulator;
            let t_affine: G2Affine = doubled.into();
            let (x_t, y_t) = if t_affine.is_zero() {
                (Fq2::zero(), Fq2::zero())
            } else {
                (t_affine.x, t_affine.y)
            };
            x_t_values.push(x_t);
            y_t_values.push(y_t);

            // Indicator for T_i = O
            let t_is_inf = if t_affine.is_zero() {
                Fq::one()
            } else {
                Fq::zero()
            };
            t_is_infinity_values.push(t_is_inf);

            // Conditional add: A_{i+1} = T_i + b_i * P
            accumulator = if bit {
                doubled.add(&point.into_group())
            } else {
                doubled
            };

            // Store A_{i+1}
            let a_next: G2Affine = accumulator.into();
            let (x_a_next, y_a_next) = if a_next.is_zero() {
                (Fq2::zero(), Fq2::zero())
            } else {
                (a_next.x, a_next.y)
            };
            x_a_values.push(x_a_next);
            y_a_values.push(y_a_next);
        }

        // Split Fq2 coords into c0/c1 over Fq
        let (x_a_c0_vals, x_a_c1_vals) = split_fq2_steps(&x_a_values[..256]);
        let (y_a_c0_vals, y_a_c1_vals) = split_fq2_steps(&y_a_values[..256]);
        let (x_t_c0_vals, x_t_c1_vals) = split_fq2_steps(&x_t_values);
        let (y_t_c0_vals, y_t_c1_vals) = split_fq2_steps(&y_t_values);

        let x_a_mle_c0 = build_mle_from_steps(&x_a_c0_vals, num_vars);
        let x_a_mle_c1 = build_mle_from_steps(&x_a_c1_vals, num_vars);
        let y_a_mle_c0 = build_mle_from_steps(&y_a_c0_vals, num_vars);
        let y_a_mle_c1 = build_mle_from_steps(&y_a_c1_vals, num_vars);

        let x_t_mle_c0 = build_mle_from_steps(&x_t_c0_vals, num_vars);
        let x_t_mle_c1 = build_mle_from_steps(&x_t_c1_vals, num_vars);
        let y_t_mle_c0 = build_mle_from_steps(&y_t_c0_vals, num_vars);
        let y_t_mle_c1 = build_mle_from_steps(&y_t_c1_vals, num_vars);

        let t_is_infinity_mle = build_mle_from_steps(&t_is_infinity_values, num_vars);
        let a_is_infinity_mle = build_mle_from_steps(&a_is_infinity_values, num_vars);
        let bit_mle = build_mle_from_steps(&bit_values, num_vars);

        // Shifted A_{i+1} MLEs: A_1..A_256
        let x_a_next_values = x_a_values[1..257].to_vec();
        let y_a_next_values = y_a_values[1..257].to_vec();
        let (x_a_next_c0_vals, x_a_next_c1_vals) = split_fq2_steps(&x_a_next_values);
        let (y_a_next_c0_vals, y_a_next_c1_vals) = split_fq2_steps(&y_a_next_values);

        let x_a_next_mle_c0 = build_mle_from_steps(&x_a_next_c0_vals, num_vars);
        let x_a_next_mle_c1 = build_mle_from_steps(&x_a_next_c1_vals, num_vars);
        let y_a_next_mle_c0 = build_mle_from_steps(&y_a_next_c0_vals, num_vars);
        let y_a_next_mle_c1 = build_mle_from_steps(&y_a_next_c1_vals, num_vars);

        let result: G2Affine = accumulator.into();

        Self {
            point_base: point,
            scalar,
            result,
            x_a_c0_mles: vec![x_a_mle_c0],
            x_a_c1_mles: vec![x_a_mle_c1],
            y_a_c0_mles: vec![y_a_mle_c0],
            y_a_c1_mles: vec![y_a_mle_c1],
            x_t_c0_mles: vec![x_t_mle_c0],
            x_t_c1_mles: vec![x_t_mle_c1],
            y_t_c0_mles: vec![y_t_mle_c0],
            y_t_c1_mles: vec![y_t_mle_c1],
            x_a_next_c0_mles: vec![x_a_next_mle_c0],
            x_a_next_c1_mles: vec![x_a_next_mle_c1],
            y_a_next_c0_mles: vec![y_a_next_mle_c0],
            y_a_next_c1_mles: vec![y_a_next_mle_c1],
            t_is_infinity_mles: vec![t_is_infinity_mle],
            a_is_infinity_mles: vec![a_is_infinity_mle],
            bit_mles: vec![bit_mle],
            bits: bits_msb,
        }
    }

    /// Verify that the result matches [scalar]point (debug helper).
    pub fn verify_result(&self) -> bool {
        let expected = self.point_base.mul_bigint(self.scalar.into_bigint());
        let expected_affine: G2Affine = expected.into();
        self.result == expected_affine
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::UniformRand;

    #[test]
    fn test_g2_scalar_multiplication_witness_result() {
        let mut rng = ark_std::test_rng();
        let point = G2Affine::rand(&mut rng);
        let scalar = Fr::rand(&mut rng);
        let witness = G2ScalarMultiplicationSteps::new(point, scalar);
        assert!(witness.verify_result());
        assert_eq!(witness.bits.len(), 256);
        assert_eq!(witness.x_a_c0_mles[0].len(), 256);
        assert_eq!(witness.x_a_c1_mles[0].len(), 256);
    }

    #[test]
    fn test_g2_scalar_multiplication_constraints_on_hypercube() {
        let mut rng = ark_std::test_rng();
        let point = G2Affine::rand(&mut rng);
        let scalar = Fr::from(55743u64);
        let witness = G2ScalarMultiplicationSteps::new(point, scalar);

        let (x_p, y_p) = (witness.point_base.x, witness.point_base.y);

        let fq2_from_fq = |v: Fq| Fq2::new(v, Fq::zero());
        let one2 = Fq2::one();

        for step in 0..256 {
            let x_a = Fq2::new(witness.x_a_c0_mles[0][step], witness.x_a_c1_mles[0][step]);
            let y_a = Fq2::new(witness.y_a_c0_mles[0][step], witness.y_a_c1_mles[0][step]);
            let x_t = Fq2::new(witness.x_t_c0_mles[0][step], witness.x_t_c1_mles[0][step]);
            let y_t = Fq2::new(witness.y_t_c0_mles[0][step], witness.y_t_c1_mles[0][step]);
            let x_a_next = Fq2::new(
                witness.x_a_next_c0_mles[0][step],
                witness.x_a_next_c1_mles[0][step],
            );
            let y_a_next = Fq2::new(
                witness.y_a_next_c0_mles[0][step],
                witness.y_a_next_c1_mles[0][step],
            );

            let ind_t = witness.t_is_infinity_mles[0][step];
            let ind_a = witness.a_is_infinity_mles[0][step];
            let bit = witness.bit_mles[0][step];

            let bit2 = fq2_from_fq(bit);
            let ind_t2 = fq2_from_fq(ind_t);

            // C1: 4y_A²(x_T + 2x_A) - 9x_A⁴ = 0
            let c1 = {
                let four = fq2_from_fq(Fq::from(4u64));
                let two = fq2_from_fq(Fq::from(2u64));
                let nine = fq2_from_fq(Fq::from(9u64));

                let y_a_sq = y_a * y_a;
                let x_a_sq = x_a * x_a;
                let x_a_fourth = x_a_sq * x_a_sq;
                four * y_a_sq * (x_t + two * x_a) - nine * x_a_fourth
            };

            // C2: 3x_A²(x_T - x_A) + 2y_A(y_T + y_A) = 0
            let c2 = {
                let three = fq2_from_fq(Fq::from(3u64));
                let two = fq2_from_fq(Fq::from(2u64));
                let x_a_sq = x_a * x_a;
                three * x_a_sq * (x_t - x_a) + two * y_a * (y_t + y_a)
            };

            // C3/C4: conditional add by bit, with T=O special case
            let c3 = {
                let c3_skip = (one2 - bit2) * (x_a_next - x_t);
                let c3_infty = bit2 * ind_t2 * (x_a_next - x_p);
                let x_diff = x_p - x_t;
                let y_diff = y_p - y_t;
                let chord_x = (x_a_next + x_t + x_p) * x_diff * x_diff - y_diff * y_diff;
                let c3_add = bit2 * (one2 - ind_t2) * chord_x;
                c3_skip + c3_infty + c3_add
            };

            let c4 = {
                let c4_skip = (one2 - bit2) * (y_a_next - y_t);
                let c4_infty = bit2 * ind_t2 * (y_a_next - y_p);
                let x_diff = x_p - x_t;
                let y_diff = y_p - y_t;
                let chord_y = (y_a_next + y_t) * x_diff - y_diff * (x_t - x_a_next);
                let c4_add = bit2 * (one2 - ind_t2) * chord_y;
                c4_skip + c4_infty + c4_add
            };

            // C5: bit booleanity
            let c5 = bit * (Fq::one() - bit);

            // C6: if A is infinity then T is infinity
            let c6 = ind_a * (Fq::one() - ind_t);

            // C7: if ind_T = 1 then x_T = y_T = 0 in Fq2
            let c7_xt_c0 = ind_t * x_t.c0;
            let c7_xt_c1 = ind_t * x_t.c1;
            let c7_yt_c0 = ind_t * y_t.c0;
            let c7_yt_c1 = ind_t * y_t.c1;

            assert!(c1.is_zero(), "C1 failed at step {}: {:?}", step, c1);
            assert!(c2.is_zero(), "C2 failed at step {}: {:?}", step, c2);
            assert!(c3.is_zero(), "C3 failed at step {}: {:?}", step, c3);
            assert!(c4.is_zero(), "C4 failed at step {}: {:?}", step, c4);
            assert!(c5.is_zero(), "C5 failed at step {}: {:?}", step, c5);
            assert!(c6.is_zero(), "C6 failed at step {}: {:?}", step, c6);
            assert!(
                c7_xt_c0.is_zero()
                    && c7_xt_c1.is_zero()
                    && c7_yt_c0.is_zero()
                    && c7_yt_c1.is_zero(),
                "C7 failed at step {}",
                step
            );
        }
    }
}
