//! Streaming-friendly interfaces for opening-reduction and MLE evaluation.
//!
//! This module is a mockup scaffolding for a future streaming refactor of the
//! Stage 5 opening-reduction (see `.notes/sumchecks_dag_spec.md`). It is not
//! wired into the prover yet and compiles standalone.

use crate::field::JoltField;

/// Abstract interface for a multilinear extension that can be evaluated and
/// bound without materializing the full coefficient vector in memory.
///
/// Implementations should compute `sumcheck_evals_array` lazily from a trace
/// and/or preprocessing (e.g. RAM/REG increments, RA one-hot providers).
pub trait StreamingMLE<F: JoltField>: Send + Sync {
    /// Total number of variables for this MLE.
    fn num_vars(&self) -> usize;

    /// Bind one variable with challenge `r_j` using the instance's native
    /// binding order (typically HighToLow). Implementations should advance
    /// internal state instead of cloning or reallocating buffers.
    fn bind(&mut self, r_j: F);

    /// Return the degree-many evaluations needed by the sumcheck prover at the
    /// half-domain index `i` for the current round (post-binding). This mirrors
    /// `MultilinearPolynomial::sumcheck_evals_array::<DEGREE>` but without
    /// materializing the polynomial.
    /// Return [eval(0), eval(2), ..., eval(DEGREE-1)] at the current round.
    /// Implementations can ignore elements beyond what the degree requires.
    fn sumcheck_evals_degree2(&self, i: usize) -> [F; 2];
}

/// A lightweight wrapper describing a random linear combination of multiple
/// streaming MLEs. This is intended for the per-instance RLC in opening
/// reduction (combining multiple polynomials opened at the same point) without
/// building a dense vector.
pub struct RlcStreamingPolynomial<F: JoltField> {
    pub coeffs: Vec<F>,
    // Each entry is an underlying streaming MLE provider.
    pub streams: Vec<Box<dyn StreamingMLE<F>>>,
}

impl<F: JoltField> RlcStreamingPolynomial<F> {
    pub fn new(coeffs: Vec<F>, streams: Vec<Box<dyn StreamingMLE<F>>>) -> Self {
        debug_assert_eq!(coeffs.len(), streams.len());
        Self { coeffs, streams }
    }

    pub fn num_vars(&self) -> usize {
        self.streams
            .first()
            .map(|s| s.num_vars())
            .unwrap_or(0)
    }

    pub fn bind(&mut self, r_j: F) {
        self.streams.iter_mut().for_each(|s| s.bind(r_j));
    }

    pub fn sumcheck_evals_degree2(&self, i: usize) -> [F; 2] {
        let mut acc = [F::zero(); 2];
        for (c, s) in self.coeffs.iter().zip(self.streams.iter()) {
            let evals = s.sumcheck_evals_degree2(i);
            acc[0] += *c * evals[0];
            acc[1] += *c * evals[1];
        }
        acc
    }
}


