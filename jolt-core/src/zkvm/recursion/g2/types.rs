//! Shared types for G2 group operations.

use crate::field::JoltField;
use crate::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};
use crate::zkvm::witness::{G2AddTerm, TermEnum};
use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use super::{fq2_mul_c0, fq2_mul_c1, fq2_sq_c0, fq2_sq_c1};

/// Public inputs for a single G2 scalar multiplication (the scalar).
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G2ScalarMulPublicInputs {
    pub scalar: Fr,
}

impl GuestSerialize for G2ScalarMulPublicInputs {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.scalar.guest_serialize(w)
    }
}

impl GuestDeserialize for G2ScalarMulPublicInputs {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            scalar: Fr::guest_deserialize(r)?,
        })
    }
}

impl G2ScalarMulPublicInputs {
    pub fn new(scalar: Fr) -> Self {
        Self { scalar }
    }

    pub fn bits_msb(&self) -> Vec<bool> {
        let scalar_bits_le = self.scalar.into_bigint().to_bits_le();
        (0..256).rev().map(|i| scalar_bits_le[i]).collect()
    }

    pub fn evaluate_bit_mle<F: JoltField>(&self, eval_point: &[F]) -> F {
        let (step_point, pad_sel) = match eval_point.len() {
            8 => (eval_point, F::one()),
            11 => {
                let (step, pad) = eval_point.split_at(8);
                let mut sel = F::one();
                let one = F::one();
                for &p_i in pad {
                    sel *= one - p_i;
                }
                (step, sel)
            }
            _ => panic!(
                "G2ScalarMulPublicInputs::evaluate_bit_mle expected 8 (native) or 11 (padded) vars, got {}",
                eval_point.len()
            ),
        };
        let bits = self.bits_msb();

        let mut evals: Vec<F> = bits
            .iter()
            .map(|&b| if b { F::one() } else { F::zero() })
            .collect();
        debug_assert_eq!(evals.len(), 256);

        let mut len = evals.len();
        for &r_i in step_point {
            let half = len / 2;
            for j in 0..half {
                let a = evals[2 * j];
                let b = evals[2 * j + 1];
                evals[j] = a + r_i * (b - a);
            }
            len = half;
        }
        debug_assert_eq!(len, 1);
        pad_sel * evals[0]
    }
}

/// Values for G2 addition (used during fused addition).
#[derive(Clone, Debug, Default)]
pub struct G2AddValues<F> {
    pub x_p_c0: F,
    pub x_p_c1: F,
    pub y_p_c0: F,
    pub y_p_c1: F,
    pub ind_p: F,
    pub x_q_c0: F,
    pub x_q_c1: F,
    pub y_q_c0: F,
    pub y_q_c1: F,
    pub ind_q: F,
    pub x_r_c0: F,
    pub x_r_c1: F,
    pub y_r_c0: F,
    pub y_r_c1: F,
    pub ind_r: F,
    pub lambda_c0: F,
    pub lambda_c1: F,
    pub inv_delta_x_c0: F,
    pub inv_delta_x_c1: F,
    pub is_double: F,
    pub is_inverse: F,
}

impl<F: JoltField> G2AddValues<F> {
    /// Construct values from a batch of per-term univariate evaluations.
    ///
    /// `poly_evals[t][eval_index]` corresponds to the `t`-th `G2AddTerm` (see `zkvm/witness.rs`)
    /// evaluated at the `eval_index`-th point (0..degree).
    pub fn from_poly_evals<const DEGREE: usize>(
        poly_evals: &[[F; DEGREE]],
        eval_index: usize,
    ) -> Self {
        debug_assert_eq!(
            poly_evals.len(),
            G2AddTerm::COUNT,
            "expected one eval array per G2AddTerm"
        );
        Self {
            x_p_c0: poly_evals[0][eval_index],
            x_p_c1: poly_evals[1][eval_index],
            y_p_c0: poly_evals[2][eval_index],
            y_p_c1: poly_evals[3][eval_index],
            ind_p: poly_evals[4][eval_index],
            x_q_c0: poly_evals[5][eval_index],
            x_q_c1: poly_evals[6][eval_index],
            y_q_c0: poly_evals[7][eval_index],
            y_q_c1: poly_evals[8][eval_index],
            ind_q: poly_evals[9][eval_index],
            x_r_c0: poly_evals[10][eval_index],
            x_r_c1: poly_evals[11][eval_index],
            y_r_c0: poly_evals[12][eval_index],
            y_r_c1: poly_evals[13][eval_index],
            ind_r: poly_evals[14][eval_index],
            lambda_c0: poly_evals[15][eval_index],
            lambda_c1: poly_evals[16][eval_index],
            inv_delta_x_c0: poly_evals[17][eval_index],
            inv_delta_x_c1: poly_evals[18][eval_index],
            is_double: poly_evals[19][eval_index],
            is_inverse: poly_evals[20][eval_index],
        }
    }

    /// Construct values from opened claims, ordered by `G2AddTerm` index.
    pub fn from_claims(claims: &[F]) -> Self {
        debug_assert_eq!(
            claims.len(),
            G2AddTerm::COUNT,
            "expected one claim per G2AddTerm"
        );
        Self {
            x_p_c0: claims[0],
            x_p_c1: claims[1],
            y_p_c0: claims[2],
            y_p_c1: claims[3],
            ind_p: claims[4],
            x_q_c0: claims[5],
            x_q_c1: claims[6],
            y_q_c0: claims[7],
            y_q_c1: claims[8],
            ind_q: claims[9],
            x_r_c0: claims[10],
            x_r_c1: claims[11],
            y_r_c0: claims[12],
            y_r_c1: claims[13],
            ind_r: claims[14],
            lambda_c0: claims[15],
            lambda_c1: claims[16],
            inv_delta_x_c0: claims[17],
            inv_delta_x_c1: claims[18],
            is_double: claims[19],
            is_inverse: claims[20],
        }
    }

    /// Evaluate the batched G2 add constraint polynomial at this point.
    ///
    /// Uses `delta` to batch the constraint terms.
    /// Fq2 arithmetic is done component-wise using helpers from the parent module.
    pub fn eval_constraint(&self, delta: F) -> F {
        let one = F::one();
        let three = F::from_u64(3);

        let dx_c0 = self.x_q_c0 - self.x_p_c0;
        let dx_c1 = self.x_q_c1 - self.x_p_c1;
        let dy_c0 = self.y_q_c0 - self.y_p_c0;
        let dy_c1 = self.y_q_c1 - self.y_p_c1;
        let s_finite = (one - self.ind_p) * (one - self.ind_q);

        let mut acc = F::zero();
        let mut delta_pow = F::one();

        // Boolean constraints for indicators
        acc += delta_pow * (self.ind_p * (one - self.ind_p));
        delta_pow *= delta;
        acc += delta_pow * (self.ind_q * (one - self.ind_q));
        delta_pow *= delta;
        acc += delta_pow * (self.ind_r * (one - self.ind_r));
        delta_pow *= delta;

        // Infinity encoding: ind * coord = 0
        acc += delta_pow * (self.ind_p * self.x_p_c0);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_p * self.x_p_c1);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_p * self.y_p_c0);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_p * self.y_p_c1);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_q * self.x_q_c0);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_q * self.x_q_c1);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_q * self.y_q_c0);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_q * self.y_q_c1);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_r * self.x_r_c0);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_r * self.x_r_c1);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_r * self.y_r_c0);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_r * self.y_r_c1);
        delta_pow *= delta;

        // If P = O then R = Q
        acc += delta_pow * (self.ind_p * (self.x_r_c0 - self.x_q_c0));
        delta_pow *= delta;
        acc += delta_pow * (self.ind_p * (self.x_r_c1 - self.x_q_c1));
        delta_pow *= delta;
        acc += delta_pow * (self.ind_p * (self.y_r_c0 - self.y_q_c0));
        delta_pow *= delta;
        acc += delta_pow * (self.ind_p * (self.y_r_c1 - self.y_q_c1));
        delta_pow *= delta;
        acc += delta_pow * (self.ind_p * (self.ind_r - self.ind_q));
        delta_pow *= delta;

        // If Q = O and P != O then R = P
        let q_inf = self.ind_q * (one - self.ind_p);
        acc += delta_pow * (q_inf * (self.x_r_c0 - self.x_p_c0));
        delta_pow *= delta;
        acc += delta_pow * (q_inf * (self.x_r_c1 - self.x_p_c1));
        delta_pow *= delta;
        acc += delta_pow * (q_inf * (self.y_r_c0 - self.y_p_c0));
        delta_pow *= delta;
        acc += delta_pow * (q_inf * (self.y_r_c1 - self.y_p_c1));
        delta_pow *= delta;
        acc += delta_pow * (q_inf * (self.ind_r - self.ind_p));
        delta_pow *= delta;

        // Booleanity of branch bits in finite case
        acc += delta_pow * (s_finite * self.is_double * (one - self.is_double));
        delta_pow *= delta;
        acc += delta_pow * (s_finite * self.is_inverse * (one - self.is_inverse));
        delta_pow *= delta;

        // Branch selection: inv_dx * dx = 1 (c0) and 0 (c1) in the generic add case
        let inv_dx_times_dx_c0 = fq2_mul_c0(self.inv_delta_x_c0, self.inv_delta_x_c1, dx_c0, dx_c1);
        let inv_dx_times_dx_c1 = fq2_mul_c1(self.inv_delta_x_c0, self.inv_delta_x_c1, dx_c0, dx_c1);
        acc += delta_pow
            * (s_finite * (one - self.is_double - self.is_inverse) * (one - inv_dx_times_dx_c0));
        delta_pow *= delta;
        acc +=
            delta_pow * (s_finite * (one - self.is_double - self.is_inverse) * inv_dx_times_dx_c1);
        delta_pow *= delta;

        // If doubling, enforce P == Q (dx = 0, dy = 0)
        acc += delta_pow * (s_finite * self.is_double * dx_c0);
        delta_pow *= delta;
        acc += delta_pow * (s_finite * self.is_double * dx_c1);
        delta_pow *= delta;
        acc += delta_pow * (s_finite * self.is_double * dy_c0);
        delta_pow *= delta;
        acc += delta_pow * (s_finite * self.is_double * dy_c1);
        delta_pow *= delta;

        // If inverse, enforce P == -Q (dx = 0, y_q + y_p = 0)
        acc += delta_pow * (s_finite * self.is_inverse * dx_c0);
        delta_pow *= delta;
        acc += delta_pow * (s_finite * self.is_inverse * dx_c1);
        delta_pow *= delta;
        acc += delta_pow * (s_finite * self.is_inverse * (self.y_q_c0 + self.y_p_c0));
        delta_pow *= delta;
        acc += delta_pow * (s_finite * self.is_inverse * (self.y_q_c1 + self.y_p_c1));
        delta_pow *= delta;

        // Slope equation (add): lambda * dx = dy
        let lam_dx_c0 = fq2_mul_c0(self.lambda_c0, self.lambda_c1, dx_c0, dx_c1);
        let lam_dx_c1 = fq2_mul_c1(self.lambda_c0, self.lambda_c1, dx_c0, dx_c1);
        acc +=
            delta_pow * (s_finite * (one - self.is_double - self.is_inverse) * (lam_dx_c0 - dy_c0));
        delta_pow *= delta;
        acc +=
            delta_pow * (s_finite * (one - self.is_double - self.is_inverse) * (lam_dx_c1 - dy_c1));
        delta_pow *= delta;

        // Slope equation (double): 2*y_p*lambda = 3*x_p^2
        let two = F::from_u64(2);
        let two_yp_lam_c0 =
            two * fq2_mul_c0(self.y_p_c0, self.y_p_c1, self.lambda_c0, self.lambda_c1);
        let two_yp_lam_c1 =
            two * fq2_mul_c1(self.y_p_c0, self.y_p_c1, self.lambda_c0, self.lambda_c1);
        let three_xp_sq_c0 = three * fq2_sq_c0(self.x_p_c0, self.x_p_c1);
        let three_xp_sq_c1 = three * fq2_sq_c1(self.x_p_c0, self.x_p_c1);
        acc += delta_pow * (s_finite * self.is_double * (two_yp_lam_c0 - three_xp_sq_c0));
        delta_pow *= delta;
        acc += delta_pow * (s_finite * self.is_double * (two_yp_lam_c1 - three_xp_sq_c1));
        delta_pow *= delta;

        // Inverse => ind_R = 1
        acc += delta_pow * (s_finite * self.is_inverse * (one - self.ind_r));
        delta_pow *= delta;
        // Non-inverse => ind_R = 0
        acc += delta_pow * (s_finite * (one - self.is_inverse) * self.ind_r);
        delta_pow *= delta;

        // x_R formula for non-inverse: x_r = lambda^2 - x_p - x_q
        let lam_sq_c0 = fq2_sq_c0(self.lambda_c0, self.lambda_c1);
        let lam_sq_c1 = fq2_sq_c1(self.lambda_c0, self.lambda_c1);
        acc += delta_pow
            * (s_finite
                * (one - self.is_inverse)
                * (self.x_r_c0 - (lam_sq_c0 - self.x_p_c0 - self.x_q_c0)));
        delta_pow *= delta;
        acc += delta_pow
            * (s_finite
                * (one - self.is_inverse)
                * (self.x_r_c1 - (lam_sq_c1 - self.x_p_c1 - self.x_q_c1)));
        delta_pow *= delta;

        // y_R formula for non-inverse: y_r = lambda*(x_p - x_r) - y_p
        let xp_minus_xr_c0 = self.x_p_c0 - self.x_r_c0;
        let xp_minus_xr_c1 = self.x_p_c1 - self.x_r_c1;
        let lam_times_diff_c0 = fq2_mul_c0(
            self.lambda_c0,
            self.lambda_c1,
            xp_minus_xr_c0,
            xp_minus_xr_c1,
        );
        let lam_times_diff_c1 = fq2_mul_c1(
            self.lambda_c0,
            self.lambda_c1,
            xp_minus_xr_c0,
            xp_minus_xr_c1,
        );
        acc += delta_pow
            * (s_finite
                * (one - self.is_inverse)
                * (self.y_r_c0 - (lam_times_diff_c0 - self.y_p_c0)));
        delta_pow *= delta;
        acc += delta_pow
            * (s_finite
                * (one - self.is_inverse)
                * (self.y_r_c1 - (lam_times_diff_c1 - self.y_p_c1)));

        acc
    }
}
