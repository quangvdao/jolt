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
        l_c0_c0, l_c0_c1, l_c1_c0, l_c1_c1, l_c2_c0, l_c2_c1,
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
        let _op_y0 = self.is_double * ty0 + self.is_add * self.y_q_c0;
        let _op_y1 = self.is_double * ty1 + self.is_add * self.y_q_c1;

        // Slope constraints
        // Double case: 2 * y * lambda = 3 * x^2
        let two_y_lam0 = two * mul_c0(ty0, ty1, self.lambda_c0, self.lambda_c1);
        let two_y_lam1 = two * (ty0 * self.lambda_c1 + ty1 * self.lambda_c0);
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
        let lam_dx1 = self.lambda_c0 * dx1 + self.lambda_c1 * dx0;

        acc += delta_pow * (self.is_add * (lam_dx0 - dy0));
        delta_pow *= delta;
        acc += delta_pow * (self.is_add * (lam_dx1 - dy1));
        delta_pow *= delta;

        // Inverse constraint for add case: inv_dx * dx = 1
        let inv_dx_dx0 = mul_c0(self.inv_delta_x_c0, self.inv_delta_x_c1, dx0, dx1);
        let inv_dx_dx1 = self.inv_delta_x_c0 * dx1 + self.inv_delta_x_c1 * dx0;

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
        let lam_dx_next1 = self.lambda_c0 * dx_next1 + self.lambda_c1 * dx_next0;

        acc += delta_pow * (is_active * (ty_next0 - (lam_dx_next0 - ty0)));
        delta_pow *= delta;
        acc += delta_pow * (is_active * (ty_next1 - (lam_dx_next1 - ty1)));
        delta_pow *= delta;

        // 3. Line Evaluation Coefficients (BN254 / TwistType::D convention)
        //
        // We use unscaled line coefficients (c0,c1,c2) ∈ Fq2 for the line:
        //   c0 * y + c1 * x + c2 = 0
        // then the pairing line contribution is embedded via:
        //   c0' = c0 * y_P,  c1' = c1 * x_P,  c2' = c2
        // and mapped into the Fq12 MLE using the selector polynomials.
        //
        // Double (tangent) at T:
        //   c0 = -2y_T
        //   c1 = 3x_T^2
        //   c2 = 2y_T^2 - 3x_T^3
        //
        // Add (chord) through T and Q:
        //   c0 = x_T - x_Q
        //   c1 = y_Q - y_T
        //   c2 = x_Q*y_T - x_T*y_Q

        // --- Double coefficients ---
        let dbl_c0_0 = -(two * ty0);
        let dbl_c0_1 = -(two * ty1);
        let x_sq0 = sq_c0(tx0, tx1);
        let x_sq1 = sq_c1(tx0, tx1);
        let dbl_c1_0 = three * x_sq0;
        let dbl_c1_1 = three * x_sq1;
        let y_sq0 = sq_c0(ty0, ty1);
        let y_sq1 = sq_c1(ty0, ty1);
        let x_cub0 = mul_c0(x_sq0, x_sq1, tx0, tx1);
        let x_cub1 = x_sq0 * tx1 + x_sq1 * tx0;
        let dbl_c2_0 = two * y_sq0 - three * x_cub0;
        let dbl_c2_1 = two * y_sq1 - three * x_cub1;

        // --- Add coefficients ---
        let add_c0_0 = tx0 - self.x_q_c0;
        let add_c0_1 = tx1 - self.x_q_c1;
        let add_c1_0 = self.y_q_c0 - ty0;
        let add_c1_1 = self.y_q_c1 - ty1;
        let xq_yt_0 = mul_c0(self.x_q_c0, self.x_q_c1, ty0, ty1);
        let xq_yt_1 = self.x_q_c0 * ty1 + self.x_q_c1 * ty0;
        let xt_yq_0 = mul_c0(tx0, tx1, self.y_q_c0, self.y_q_c1);
        let xt_yq_1 = tx0 * self.y_q_c1 + tx1 * self.y_q_c0;
        let add_c2_0 = xq_yt_0 - xt_yq_0;
        let add_c2_1 = xq_yt_1 - xt_yq_1;

        // Select coefficients based on branch.
        let c0_0 = self.is_double * dbl_c0_0 + self.is_add * add_c0_0;
        let c0_1 = self.is_double * dbl_c0_1 + self.is_add * add_c0_1;
        let c1_0 = self.is_double * dbl_c1_0 + self.is_add * add_c1_0;
        let c1_1 = self.is_double * dbl_c1_1 + self.is_add * add_c1_1;
        let c2_0 = self.is_double * dbl_c2_0 + self.is_add * add_c2_0;
        let c2_1 = self.is_double * dbl_c2_1 + self.is_add * add_c2_1;

        // Constrain witnessed coefficients.
        acc += delta_pow * (is_active * (self.l_c0_c0 - c0_0));
        delta_pow *= delta;
        acc += delta_pow * (is_active * (self.l_c0_c1 - c0_1));
        delta_pow *= delta;
        acc += delta_pow * (is_active * (self.l_c1_c0 - c1_0));
        delta_pow *= delta;
        acc += delta_pow * (is_active * (self.l_c1_c1 - c1_1));
        delta_pow *= delta;
        acc += delta_pow * (is_active * (self.l_c2_c0 - c2_0));
        delta_pow *= delta;
        acc += delta_pow * (is_active * (self.l_c2_c1 - c2_1));
        delta_pow *= delta;

        // 4. Line Evaluation Value (sparse 034 embedding)
        //
        // We embed the Fq2 coefficients into an Fq12 element with nonzero coefficients
        // at positions (0,3,4) and then use selector polynomials to evaluate its MLE at x:
        //   coeff0 = c0 * y_P   (Fq2)
        //   coeff3 = c1 * x_P   (Fq2)
        //   coeff4 = c2         (Fq2)
        let coeff0_c0 = self.l_c0_c0 * self.y_p;
        let coeff0_c1 = self.l_c0_c1 * self.y_p;
        let coeff3_c0 = self.l_c1_c0 * self.x_p;
        let coeff3_c1 = self.l_c1_c1 * self.x_p;
        let coeff4_c0 = self.l_c2_c0;
        let coeff4_c1 = self.l_c2_c1;

        let calc_l_val = self.selector_0 * coeff0_c0
            + self.selector_1 * coeff0_c1
            + self.selector_2 * coeff3_c0
            + self.selector_3 * coeff3_c1
            + self.selector_4 * coeff4_c0
            + self.selector_5 * coeff4_c1;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::commitment::dory::witness::multi_miller_loop::MultiMillerLoopSteps;
    use ark_bn254::{Fq, G1Affine, G2Affine};
    use ark_ec::AffineRepr;
    use ark_ff::Zero;

    #[test]
    fn test_constraint_zero_on_boolean_hypercube_single_pair() {
        let p = G1Affine::generator();
        let q = G2Affine::generator();
        let steps = MultiMillerLoopSteps::new(&[p], &[q]);

        let pair = 0usize;
        let f = &steps.f_packed_mles[pair];
        let f_next = &steps.f_next_packed_mles[pair];
        let quotient = &steps.quotient_packed_mles[pair];
        let t_x_c0 = &steps.t_x_c0_packed_mles[pair];
        let t_x_c1 = &steps.t_x_c1_packed_mles[pair];
        let t_y_c0 = &steps.t_y_c0_packed_mles[pair];
        let t_y_c1 = &steps.t_y_c1_packed_mles[pair];
        let t_x_c0_next = &steps.t_x_c0_next_packed_mles[pair];
        let t_x_c1_next = &steps.t_x_c1_next_packed_mles[pair];
        let t_y_c0_next = &steps.t_y_c0_next_packed_mles[pair];
        let t_y_c1_next = &steps.t_y_c1_next_packed_mles[pair];
        let lambda_c0 = &steps.lambda_c0_packed_mles[pair];
        let lambda_c1 = &steps.lambda_c1_packed_mles[pair];
        let inv_dx_c0 = &steps.inv_dx_c0_packed_mles[pair];
        let inv_dx_c1 = &steps.inv_dx_c1_packed_mles[pair];
        let l_c0_c0 = &steps.l_c0_c0_packed_mles[pair];
        let l_c0_c1 = &steps.l_c0_c1_packed_mles[pair];
        let l_c1_c0 = &steps.l_c1_c0_packed_mles[pair];
        let l_c1_c1 = &steps.l_c1_c1_packed_mles[pair];
        let l_c2_c0 = &steps.l_c2_c0_packed_mles[pair];
        let l_c2_c1 = &steps.l_c2_c1_packed_mles[pair];
        let x_p = &steps.x_p_packed_mles[pair];
        let y_p = &steps.y_p_packed_mles[pair];
        let x_q_c0 = &steps.x_q_c0_packed_mles[pair];
        let x_q_c1 = &steps.x_q_c1_packed_mles[pair];
        let y_q_c0 = &steps.y_q_c0_packed_mles[pair];
        let y_q_c1 = &steps.y_q_c1_packed_mles[pair];
        let is_double = &steps.is_double_packed_mles[pair];
        let is_add = &steps.is_add_packed_mles[pair];
        let l_val = &steps.l_val_packed_mles[pair];
        let g = &steps.g_packed_mles[pair];
        let selector_0 = &steps.selector_0_packed_mles[pair];
        let selector_1 = &steps.selector_1_packed_mles[pair];
        let selector_2 = &steps.selector_2_packed_mles[pair];
        let selector_3 = &steps.selector_3_packed_mles[pair];
        let selector_4 = &steps.selector_4_packed_mles[pair];
        let selector_5 = &steps.selector_5_packed_mles[pair];
        let delta1 = Fq::from(7u64);
        let delta2 = Fq::from(13u64);

        let step_size = 1usize << 7; // 128
        let elem_size = 1usize << 4; // 16

        for s in 0..step_size {
            for x in 0..elem_size {
                let idx = x * step_size + s;

                let vals = MultiMillerLoopValues::<Fq> {
                    f: f[idx],
                    f_next: f_next[idx],
                    quotient: quotient[idx],
                    t_x_c0: t_x_c0[idx],
                    t_x_c1: t_x_c1[idx],
                    t_y_c0: t_y_c0[idx],
                    t_y_c1: t_y_c1[idx],
                    t_x_c0_next: t_x_c0_next[idx],
                    t_x_c1_next: t_x_c1_next[idx],
                    t_y_c0_next: t_y_c0_next[idx],
                    t_y_c1_next: t_y_c1_next[idx],
                    lambda_c0: lambda_c0[idx],
                    lambda_c1: lambda_c1[idx],
                    inv_delta_x_c0: inv_dx_c0[idx],
                    inv_delta_x_c1: inv_dx_c1[idx],
                    l_c0_c0: l_c0_c0[idx],
                    l_c0_c1: l_c0_c1[idx],
                    l_c1_c0: l_c1_c0[idx],
                    l_c1_c1: l_c1_c1[idx],
                    l_c2_c0: l_c2_c0[idx],
                    l_c2_c1: l_c2_c1[idx],
                    x_p: x_p[idx],
                    y_p: y_p[idx],
                    x_q_c0: x_q_c0[idx],
                    x_q_c1: x_q_c1[idx],
                    y_q_c0: y_q_c0[idx],
                    y_q_c1: y_q_c1[idx],
                    is_double: is_double[idx],
                    is_add: is_add[idx],
                    l_val: l_val[idx],
                    g: g[idx],
                    selector_0: selector_0[idx],
                    selector_1: selector_1[idx],
                    selector_2: selector_2[idx],
                    selector_3: selector_3[idx],
                    selector_4: selector_4[idx],
                    selector_5: selector_5[idx],
                };

                let c1 = vals.eval_constraint(delta1);
                let c2 = vals.eval_constraint(delta2);
                if !c1.is_zero() || !c2.is_zero() {
                    panic!(
                        "constraint nonzero at (s={s}, x={x}): c1={c1:?}, c2={c2:?}, is_double={}, is_add={}",
                        vals.is_double, vals.is_add
                    );
                }
            }
        }
    }
}
