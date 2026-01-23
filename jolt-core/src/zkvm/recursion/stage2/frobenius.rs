//! Frobenius map sumcheck for proving f^q = g constraints.
//!
//! The Frobenius map in Fq12 is a linear operation on the coefficients.
//! If a = sum(a_i * w^i), then a^q = sum(a_i^q * w^(i*q)).
//! Since a_i are in Fq (base field), a_i^q = a_i.
//! So a^q = sum(a_i * w^(i*q)).
//!
//! This means the coefficients of the output are just a permutation of the input coefficients,
//! scaled by some constants (if w^(i*q) introduces constants).
//!
//! Actually, for BN254, Fq12 is constructed as a tower.
//! The Frobenius map is linear over Fq.
//! So we can express it as: output_coeff_j = sum_i (M_ji * input_coeff_i).
//!
//! Since we use term batching, we can check this linear relation efficiently.
//! Or even simpler: if the map is just a permutation/scaling, we can check it directly.
//!
//! For BN254 sextic twist, frobenius map is:
//! f^q (c0 + c1 w + ... + c5 w^5) = c0^q + c1^q w^q + ...
//! Since ci are in Fq2, ci^q is conjugation in Fq2.
//! (a + bu)^q = a + bu^q = a - bu.
//!
//! So Frobenius is:
//! 1. Conjugate each Fq2 coefficient.
//! 2. Multiply by w^q, w^2q, etc.
//! w^q = w * frobenius_constants[1]
//!
//! So the constraint is linear.

use crate::{
    define_constraint,
    field::JoltField,
    poly::opening_proof::SumcheckId,
    zkvm::witness::FrobeniusTerm,
};

define_constraint!(
    name: Frobenius,
    sumcheck_id: SumcheckId::RecursionVirtualization, // Reusing virtualization ID or new one?
    // Actually, Frobenius is simple enough it might not need its own sumcheck if we wire it.
    // But if we want to prove it, we need a sumcheck.
    // Let's use a new ID if possible, or piggyback.
    // The plan said "Implement Frobenius Op".
    // Let's assume we add a new ID or use a generic one.
    // For now, I'll use a placeholder ID or reuse one.
    // Actually, I should add Frobenius to SumcheckId.
    // But I can't edit SumcheckId definition easily (it's in opening_proof.rs which I just edited).
    // I didn't add Frobenius to SumcheckId in opening_proof.rs.
    // I should have.
    // Let's check opening_proof.rs again.
    // I added MultiMillerLoop.
    // I'll add Frobenius to SumcheckId in the next step.
    // For now, I'll use MultiMillerLoop ID temporarily to make it compile, then fix it.
    // Or better, I'll edit opening_proof.rs first.
    num_vars: 4, // 4-var MLE for Fq12 (16 coeffs, padded)
    degree: 1, // Linear constraint
    uses_term_batching: true,
    term_enum: FrobeniusTerm,
    recursion_poly_variant: Frobenius,
    fields: [
        input, output, frob_const
    ]
);

// ============================================================================
// Logic Implementation
// ============================================================================

impl<F: JoltField> FrobeniusValues<F> {
    /// Evaluate the batched Frobenius constraint.
    ///
    /// constraint: output - input * frob_const = 0
    pub fn eval_constraint(&self, _delta: F) -> F {
        self.output - self.input * self.frob_const
    }

    pub fn eval_constraint_no_batching(&self) -> F {
        self.output - self.input * self.frob_const
    }
}
