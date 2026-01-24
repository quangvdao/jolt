//! G2 addition sumcheck for proving G2 group addition constraints.
//!
//! This is the G2 analogue of `g1_add.rs`, but for points over Fq2. Since the recursion SNARK
//! runs over the base field Fq, we split each Fq2 coordinate into (c0,c1) components in Fq and
//! enforce all constraints component-wise.
//!
//! This protocol uses the generic ConstraintListSumcheck wrapper with term batching.

use crate::{
    define_constraint, field::JoltField, poly::opening_proof::SumcheckId, zkvm::witness::G2AddTerm,
};

use super::{fq2_mul_c0, fq2_mul_c1, fq2_sq_c0, fq2_sq_c1};

define_constraint!(
    name: G2Add,
    sumcheck_id: SumcheckId::G2Add,
    num_vars: 11,
    degree: 6,
    uses_term_batching: true,
    term_enum: G2AddTerm,
    recursion_poly_variant: G2Add,
    fields: [
        x_p_c0, x_p_c1, y_p_c0, y_p_c1, ind_p,
        x_q_c0, x_q_c1, y_q_c0, y_q_c1, ind_q,
        x_r_c0, x_r_c1, y_r_c0, y_r_c1, ind_r,
        lambda_c0, lambda_c1, inv_delta_x_c0, inv_delta_x_c1,
        is_double, is_inverse
    ]
);

// ============================================================================
// Logic Implementation
// ============================================================================

impl<F: JoltField> G2AddValues<F> {
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
        let inv_dx_times_dx_c0 =
            fq2_mul_c0(self.inv_delta_x_c0, self.inv_delta_x_c1, dx_c0, dx_c1);
        let inv_dx_times_dx_c1 =
            fq2_mul_c1(self.inv_delta_x_c0, self.inv_delta_x_c1, dx_c0, dx_c1);
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
        let lam_times_diff_c0 =
            fq2_mul_c0(self.lambda_c0, self.lambda_c1, xp_minus_xr_c0, xp_minus_xr_c1);
        let lam_times_diff_c1 =
            fq2_mul_c1(self.lambda_c0, self.lambda_c1, xp_minus_xr_c0, xp_minus_xr_c1);
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

    pub fn eval_constraint_no_batching(&self) -> F {
        unreachable!("G2Add uses term batching")
    }
}
