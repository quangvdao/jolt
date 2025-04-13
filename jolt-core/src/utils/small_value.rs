use crate::field::JoltField;

// Helper functions for small value sumcheck

// The coefficients for the randomness, used in Algorithm 4
pub fn r_coeffs<F: JoltField>(r: F) -> [F; 3] {
    [r.square(), (F::one() - r).square(), r * (F::one() - r)]
}
