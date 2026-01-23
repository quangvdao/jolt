//! Multi-Miller loop sumcheck for proving pairing constraints.
//!
//! This sumcheck verifies the Multi-Miller loop computation for BN254 pairings.
//! It combines:
//! 1. G2 point updates (affine coordinates with witnessed inverses)
//! 2. Line evaluations (ell-coefficients)
//! 3. Accumulator updates (Fq12 multiplication via ring-switching)
//!
//! The constraints are batched using `delta` for term batching.

use crate::{
    define_constraint, field::JoltField, poly::opening_proof::SumcheckId,
    zkvm::witness::MultiMillerLoopTerm,
};

define_constraint!(
    name: MultiMillerLoop,
    sumcheck_id: SumcheckId::MultiMillerLoop,
    num_vars: 11, // 11-var packed format (7 step + 4 element)
    degree: 6, // Degree bound
    uses_term_batching: true,
    term_enum: MultiMillerLoopTerm,
    recursion_poly_variant: MultiMillerLoop,
    fields: [
        f, f_next, quotient,
        t_x_c0, t_x_c1, t_y_c0, t_y_c1,
        t_x_c0_next, t_x_c1_next, t_y_c0_next, t_y_c1_next,
        lambda_c0, lambda_c1,
        inv_delta_x_c0, inv_delta_x_c1,
        l_c0_c0, l_c0_c1, l_c1_c0, l_c1_c1,
        x_p, y_p,
        x_q_c0, x_q_c1, y_q_c0, y_q_c1,
        is_double, is_add,
        l_val, g,
        selector_0, selector_1, selector_2, selector_3, selector_4, selector_5
    ]
);

// ============================================================================
// Logic Implementation
// ============================================================================

impl<F: JoltField> MultiMillerLoopValues<F> {
    /// Evaluate the batched Multi-Miller loop constraint polynomial at this point.
    ///
    /// Uses `delta` to batch the constraint terms.
    ///
    /// Constraints:
    /// 1. G2 Arithmetic (Affine)
    /// 2. Line Evaluation (Ell-Coefficients)
    /// 3. Accumulator Update (Ring-Switching)
    pub fn eval_constraint(&self, delta: F) -> F {
        let one = F::one();
        let two = F::from_u64(2);
        let three = F::from_u64(3);

        // Fq2 helper functions (inline)
        // mul: (a0,a1) * (b0,b1) = (a0*b0 - a1*b1, a0*b1 + a1*b0)
        let mul_c0 = |a0: F, a1: F, b0: F, b1: F| a0 * b0 - a1 * b1;
        let mul_c1 = |a0: F, a1: F, b0: F, b1: F| a0 * b1 + a1 * b0;
        // sq: (a0,a1)^2 = (a0^2 - a1^2, 2*a0*a1)
        let sq_c0 = |a0: F, a1: F| a0 * a0 - a1 * a1;
        let sq_c1 = |a0: F, a1: F| two * a0 * a1;

        let mut acc = F::zero();
        let mut delta_pow = F::one();

        // 1. Booleanity of selectors
        acc += delta_pow * (self.is_double * (one - self.is_double));
        delta_pow *= delta;
        acc += delta_pow * (self.is_add * (one - self.is_add));
        delta_pow *= delta;
        // Mutually exclusive
        acc += delta_pow * (self.is_double * self.is_add);
        delta_pow *= delta;

        let is_active = self.is_double + self.is_add; // 1 if active step

        // 2. G2 Arithmetic
        // Current point T
        let tx0 = self.t_x_c0;
        let tx1 = self.t_x_c1;
        let ty0 = self.t_y_c0;
        let ty1 = self.t_y_c1;

        // Next point T_next
        let tx_next0 = self.t_x_c0_next;
        let tx_next1 = self.t_x_c1_next;
        let ty_next0 = self.t_y_c0_next;
        let ty_next1 = self.t_y_c1_next;

        // Operand point (T for double, Q for add)
        let op_x0 = self.is_double * tx0 + self.is_add * self.x_q_c0;
        let op_x1 = self.is_double * tx1 + self.is_add * self.x_q_c1;
        let op_y0 = self.is_double * ty0 + self.is_add * self.y_q_c0;
        let op_y1 = self.is_double * ty1 + self.is_add * self.y_q_c1;

        // Slope constraints
        // Double case: 2 * y * lambda = 3 * x^2
        let two_y_lam0 = two * mul_c0(ty0, ty1, self.lambda_c0, self.lambda_c1);
        let two_y_lam1 = two * mul_c1(ty0, ty1, self.lambda_c0, self.lambda_c1);
        let three_x_sq0 = three * sq_c0(tx0, tx1);
        let three_x_sq1 = three * sq_c1(tx0, tx1);

        acc += delta_pow * (self.is_double * (two_y_lam0 - three_x_sq0));
        delta_pow *= delta;
        acc += delta_pow * (self.is_double * (two_y_lam1 - three_x_sq1));
        delta_pow *= delta;

        // Add case: lambda * (x_q - x) = y_q - y
        let dx0 = self.x_q_c0 - tx0;
        let dx1 = self.x_q_c1 - tx1;
        let dy0 = self.y_q_c0 - ty0;
        let dy1 = self.y_q_c1 - ty1;

        let lam_dx0 = mul_c0(self.lambda_c0, self.lambda_c1, dx0, dx1);
        let lam_dx1 = mul_c1(self.lambda_c0, self.lambda_c1, dx0, dx1);

        acc += delta_pow * (self.is_add * (lam_dx0 - dy0));
        delta_pow *= delta;
        acc += delta_pow * (self.is_add * (lam_dx1 - dy1));
        delta_pow *= delta;

        // Inverse constraint for add case: inv_dx * dx = 1
        let inv_dx_dx0 = mul_c0(self.inv_delta_x_c0, self.inv_delta_x_c1, dx0, dx1);
        let inv_dx_dx1 = mul_c1(self.inv_delta_x_c0, self.inv_delta_x_c1, dx0, dx1);

        acc += delta_pow * (self.is_add * (inv_dx_dx0 - one));
        delta_pow *= delta;
        acc += delta_pow * (self.is_add * inv_dx_dx1);
        delta_pow *= delta;

        // Point update constraints
        // x_next = lambda^2 - x - x_op
        let lam_sq0 = sq_c0(self.lambda_c0, self.lambda_c1);
        let lam_sq1 = sq_c1(self.lambda_c0, self.lambda_c1);

        acc += delta_pow * (is_active * (tx_next0 - (lam_sq0 - tx0 - op_x0)));
        delta_pow *= delta;
        acc += delta_pow * (is_active * (tx_next1 - (lam_sq1 - tx1 - op_x1)));
        delta_pow *= delta;

        // y_next = lambda * (x - x_next) - y
        let dx_next0 = tx0 - tx_next0;
        let dx_next1 = tx1 - tx_next1;
        let lam_dx_next0 = mul_c0(self.lambda_c0, self.lambda_c1, dx_next0, dx_next1);
        let lam_dx_next1 = mul_c1(self.lambda_c0, self.lambda_c1, dx_next0, dx_next1);

        acc += delta_pow * (is_active * (ty_next0 - (lam_dx_next0 - ty0)));
        delta_pow *= delta;
        acc += delta_pow * (is_active * (ty_next1 - (lam_dx_next1 - ty1)));
        delta_pow *= delta;

        // 3. Line Evaluation Coefficients
        // c0 = lambda * x_T - y_T (for both double and add)
        // c1 = -lambda
        // c2 = 1 (implicit)

        let calc_c0_0 = mul_c0(self.lambda_c0, self.lambda_c1, tx0, tx1) - ty0;
        let calc_c0_1 = mul_c1(self.lambda_c0, self.lambda_c1, tx0, tx1) - ty1;

        acc += delta_pow * (is_active * (self.l_c0_c0 - calc_c0_0));
        delta_pow *= delta;
        acc += delta_pow * (is_active * (self.l_c0_c1 - calc_c0_1));
        delta_pow *= delta;

        acc += delta_pow * (is_active * (self.l_c1_c0 + self.lambda_c0));
        delta_pow *= delta;
        acc += delta_pow * (is_active * (self.l_c1_c1 + self.lambda_c1));
        delta_pow *= delta;

        // 4. Line Evaluation Value
        // l_val = selector_0 * c0.0 + selector_1 * c0.1
        //       + selector_2 * c1.0 * xp + selector_3 * c1.1 * xp
        //       + selector_4 * yp (assuming c2=1, c2.0=1, c2.1=0)

        let term_c0 = self.selector_0 * self.l_c0_c0 + self.selector_1 * self.l_c0_c1;
        let term_c1 = (self.selector_2 * self.l_c1_c0 + self.selector_3 * self.l_c1_c1) * self.x_p;
        let term_c2 = self.selector_4 * self.y_p;

        let calc_l_val = term_c0 + term_c1 + term_c2;

        acc += delta_pow * (is_active * (self.l_val - calc_l_val));
        delta_pow *= delta;

        // 5. Accumulator Update
        // f_next = f^2 * l_val (if double)
        // f_next = f * l_val (if add)
        // Ring switching: A * B - C - Q * g = 0

        let a = self.is_double * self.f * self.f + self.is_add * self.f;
        let b = self.l_val;
        let c = self.f_next;

        acc += delta_pow * (is_active * (a * b - c - self.quotient * self.g));
        // delta_pow *= delta; // Last term

        acc
    }

    pub fn eval_constraint_no_batching(&self) -> F {
        unreachable!("MultiMillerLoop uses term batching")
    }
}
