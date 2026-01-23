//! G1 addition sumcheck for proving G1 group addition constraints.
//!
//! Proves: 0 = Σ_x eq(eq_point, x) * Σ_i γ^i * (Σ_j δ^j * C_{i,j}(x))
//! where C_{i,j} are the per-instance addition constraints.
//!
//! This protocol uses the generic ConstraintListSumcheck wrapper with term batching.

use crate::{
    define_constraint, field::JoltField, poly::opening_proof::SumcheckId, zkvm::witness::G1AddTerm,
};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

define_constraint!(
    name: G1Add,
    sumcheck_id: SumcheckId::G1Add,
    num_vars: 11,
    degree: 6,
    uses_term_batching: true,
    term_enum: G1AddTerm,
    recursion_poly_variant: G1Add,
    fields: [
        x_p, y_p, ind_p,
        x_q, y_q, ind_q,
        x_r, y_r, ind_r,
        lambda, inv_delta_x, is_double, is_inverse
    ]
);

// ============================================================================
// Logic Implementation
// ============================================================================

impl<F: JoltField> G1AddValues<F> {
    /// Evaluate the batched G1 add constraint polynomial at this point.
    ///
    /// Uses `delta` to batch the 27 constraint terms: Σ_j δ^j * C_j
    ///
    /// This is THE core constraint logic - everything else is mechanical.
    pub fn eval_constraint(&self, delta: F) -> F {
        let one = F::one();
        let two = F::from_u64(2);
        let three = F::from_u64(3);

        let dx = self.x_q - self.x_p;
        let dy = self.y_q - self.y_p;
        let s_finite = (one - self.ind_p) * (one - self.ind_q);

        // Batch all terms with powers of δ: Σ_j δ^j * term_j
        let mut acc = F::zero();
        let mut delta_pow = F::one();

        // (0) ind_P boolean
        acc += delta_pow * (self.ind_p * (one - self.ind_p));
        delta_pow *= delta;
        // (1) ind_Q boolean
        acc += delta_pow * (self.ind_q * (one - self.ind_q));
        delta_pow *= delta;
        // (2) ind_R boolean
        acc += delta_pow * (self.ind_r * (one - self.ind_r));
        delta_pow *= delta;

        // (3..8) infinity encoding: ind * x = 0, ind * y = 0
        acc += delta_pow * (self.ind_p * self.x_p);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_p * self.y_p);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_q * self.x_q);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_q * self.y_q);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_r * self.x_r);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_r * self.y_r);
        delta_pow *= delta;

        // (9..11) if P = O then R = Q
        acc += delta_pow * (self.ind_p * (self.x_r - self.x_q));
        delta_pow *= delta;
        acc += delta_pow * (self.ind_p * (self.y_r - self.y_q));
        delta_pow *= delta;
        acc += delta_pow * (self.ind_p * (self.ind_r - self.ind_q));
        delta_pow *= delta;

        // (12..14) if Q = O and P != O then R = P
        let q_inf = self.ind_q * (one - self.ind_p);
        acc += delta_pow * (q_inf * (self.x_r - self.x_p));
        delta_pow *= delta;
        acc += delta_pow * (q_inf * (self.y_r - self.y_p));
        delta_pow *= delta;
        acc += delta_pow * (q_inf * (self.ind_r - self.ind_p));
        delta_pow *= delta;

        // (15..16) booleanity of branch bits in finite case
        acc += delta_pow * (s_finite * self.is_double * (one - self.is_double));
        delta_pow *= delta;
        acc += delta_pow * (s_finite * self.is_inverse * (one - self.is_inverse));
        delta_pow *= delta;

        // (17) branch selection: if x_Q = x_P then must be in (double or inverse),
        // else inv_dx must be the inverse of dx (so inv_dx * dx = 1).
        acc += delta_pow
            * (s_finite * (one - self.is_double - self.is_inverse) * (one - self.inv_delta_x * dx));
        delta_pow *= delta;

        // (18..19) if doubling, enforce P == Q
        acc += delta_pow * (s_finite * self.is_double * dx);
        delta_pow *= delta;
        acc += delta_pow * (s_finite * self.is_double * (self.y_q - self.y_p));
        delta_pow *= delta;

        // (20..21) if inverse, enforce P == -Q
        acc += delta_pow * (s_finite * self.is_inverse * dx);
        delta_pow *= delta;
        acc += delta_pow * (s_finite * self.is_inverse * (self.y_q + self.y_p));
        delta_pow *= delta;

        // (22) slope equation (add or double). Inverse case is ungated (vanishes).
        let add_branch = (one - self.is_double - self.is_inverse) * (dx * self.lambda - dy);
        let dbl_branch =
            self.is_double * (two * self.y_p * self.lambda - three * self.x_p * self.x_p);
        acc += delta_pow * (s_finite * (add_branch + dbl_branch));
        delta_pow *= delta;

        // (23) inverse => ind_R = 1
        acc += delta_pow * (s_finite * self.is_inverse * (one - self.ind_r));
        delta_pow *= delta;
        // (24) non-inverse => ind_R = 0
        acc += delta_pow * (s_finite * (one - self.is_inverse) * self.ind_r);
        delta_pow *= delta;

        // (25) x_R formula for non-inverse
        acc += delta_pow
            * (s_finite
                * (one - self.is_inverse)
                * (self.x_r - (self.lambda * self.lambda - self.x_p - self.x_q)));
        delta_pow *= delta;
        // (26) y_R formula for non-inverse
        acc += delta_pow
            * (s_finite
                * (one - self.is_inverse)
                * (self.y_r - (self.lambda * (self.x_p - self.x_r) - self.y_p)));

        acc
    }

    pub fn eval_constraint_no_batching(&self) -> F {
        unreachable!("G1Add uses term batching")
    }
}

// ============================================================================
// Public Inputs
// ============================================================================

/// Public inputs for a single G1 addition.
/// There are no public inputs for this sumcheck: all operands/results are witness polynomials.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G1AddPublicInputs {}
