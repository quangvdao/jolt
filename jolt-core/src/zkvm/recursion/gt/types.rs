//! Shared types for GT group operations.

use crate::field::JoltField;
use crate::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};
use crate::zkvm::recursion::constraints::config::CONFIG;
use ark_bn254::{Fq, Fq12};
use ark_ff::{One, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_optimizations::{fq12_to_poly12_coeffs, get_g_mle};

/// Number of step variables (7 for 128 base-4 steps)
pub const NUM_STEP_VARS: usize = CONFIG.step_vars;

/// Number of element variables (4 for Fq12 = 12 field elements, but we use 16 for power of 2)
pub const NUM_ELEMENT_VARS: usize = CONFIG.element_vars;

/// Total variables = step + element
pub const NUM_TOTAL_VARS: usize = CONFIG.packed_vars;

/// Public inputs for a single packed GT exponentiation.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct GtExpPublicInputs {
    /// Base GT element (Fq12) for this exponentiation
    pub base: Fq12,
    /// Scalar bits (MSB-first, no leading zeros)
    pub scalar_bits: Vec<bool>,
}

impl GuestSerialize for GtExpPublicInputs {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.base.guest_serialize(w)?;
        self.scalar_bits.guest_serialize(w)?;
        Ok(())
    }
}

impl GuestDeserialize for GtExpPublicInputs {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            base: Fq12::guest_deserialize(r)?,
            scalar_bits: Vec::<bool>::guest_deserialize(r)?,
        })
    }
}

impl GtExpPublicInputs {
    /// Create new public inputs for a GT exponentiation
    pub fn new(base: Fq12, scalar_bits: Vec<bool>) -> Self {
        Self { base, scalar_bits }
    }

    /// Evaluate both digit MLEs using pre-computed eq_evals.
    pub fn evaluate_digit_mles<F: JoltField>(&self, eq_evals: &[F]) -> (F, F) {
        debug_assert_eq!(eq_evals.len(), 1 << NUM_STEP_VARS);
        let mut digit_lo = F::zero();
        let mut digit_hi = F::zero();

        // Avoid allocating `digits_from_bits_msb(self.scalar_bits)` (hot in guest verifier).
        //
        // `digits_from_bits_msb` semantics:
        // - If bit-length is odd, pad a leading 0 bit.
        // - Then group into 2-bit digits (MSB-first): (hi, lo).
        let n = self.scalar_bits.len();
        if n == 0 {
            return (digit_lo, digit_hi);
        }
        let odd = (n & 1) == 1;
        let digits_len = (n + 1) / 2;
        let limit = digits_len.min(eq_evals.len());
        for s in 0..limit {
            let (hi, lo) = if !odd {
                // Exact pairs.
                (self.scalar_bits[2 * s], self.scalar_bits[2 * s + 1])
            } else if s == 0 {
                // Leading pad bit.
                (false, self.scalar_bits[0])
            } else {
                // Shifted by one due to pad.
                (self.scalar_bits[2 * s - 1], self.scalar_bits[2 * s])
            };
            let eq = eq_evals[s];
            if lo {
                digit_lo += eq;
            }
            if hi {
                digit_hi += eq;
            }
        }

        (digit_lo, digit_hi)
    }

    /// Evaluate base, base^2, and base^3 MLEs using pre-computed eq_evals.
    pub fn evaluate_base_powers_mle(&self, eq_evals: &[Fq]) -> (Fq, Fq, Fq) {
        debug_assert_eq!(eq_evals.len(), 1 << NUM_ELEMENT_VARS);

        // Fast path: avoid materializing the 16-point "MLE" table for the Fq12 element, and avoid
        // doing Fq12 multiplication in the verifier.
        //
        // In `jolt-optimizations`, `fq12_to_multilinear_evals(gt)` is defined as:
        // - map gt -> degree-11 polynomial coeffs in a tower basis (`fq12_to_poly12_coeffs`), then
        // - evaluate that polynomial at x = 0..15, producing 16 values (interpreted as a 4-var MLE).
        //
        // Evaluating the 4-var MLE at `r_elem` is therefore:
        //   Σ_{i=0}^{15} eq_evals[i] * poly(i).
        // We can compute this as:
        //   Σ_{k=0}^{11} coeff[k] * (Σ_{i=0}^{15} eq_evals[i] * i^k),
        // using a single precomputed power-sum vector per `r_elem`.
        //
        // For base^2/base^3 we compute polynomial multiplication in Fq[w]/(g(w)) where
        // g(w) = w^12 - 18 w^6 + 82, which matches the tower representation in `fq12_to_poly12_coeffs`.
        let power_sums = poly12_power_sums_at_eq(eq_evals);

        let base_coeffs = fq12_to_poly12_coeffs(&self.base);
        let base2_coeffs = poly12_mul_mod_g(&base_coeffs, &base_coeffs);
        let base3_coeffs = poly12_mul_mod_g(&base2_coeffs, &base_coeffs);

        let base_eval = poly12_eval_with_power_sums(&base_coeffs, &power_sums);
        let base2_eval = poly12_eval_with_power_sums(&base2_coeffs, &power_sums);
        let base3_eval = poly12_eval_with_power_sums(&base3_coeffs, &power_sums);
        (base_eval, base2_eval, base3_eval)
    }
}

/// Compute \(S_k = \sum_{i=0}^{15} eq[i] \cdot i^k\) for k=0..11 (degree-11 poly coefficients).
#[inline]
fn poly12_power_sums_at_eq(eq_evals: &[Fq]) -> [Fq; 12] {
    debug_assert_eq!(eq_evals.len(), 16);
    let mut s = [Fq::zero(); 12];
    for (i, &w) in eq_evals.iter().enumerate() {
        let x = Fq::from(i as u64);
        let mut pow = Fq::one();
        for k in 0..12 {
            s[k] += w * pow;
            pow *= x;
        }
    }
    s
}

#[inline]
fn poly12_eval_with_power_sums(coeffs: &[Fq; 12], power_sums: &[Fq; 12]) -> Fq {
    let mut acc = Fq::zero();
    for k in 0..12 {
        acc += coeffs[k] * power_sums[k];
    }
    acc
}

/// Multiply two degree-11 polynomials modulo \(g(w) = w^{12} - 18 w^6 + 82\).
#[inline]
fn poly12_mul_mod_g(a: &[Fq; 12], b: &[Fq; 12]) -> [Fq; 12] {
    // Schoolbook multiply: degree <= 22.
    let mut tmp = [Fq::zero(); 23];
    for i in 0..12 {
        for j in 0..12 {
            tmp[i + j] += a[i] * b[j];
        }
    }

    // Reduce high-degree terms using:
    //   w^12 = 18 w^6 - 82
    // derived from g(w) = w^12 - 18 w^6 + 82 = 0.
    let c18 = Fq::from(18u64);
    let c82 = Fq::from(82u64);

    // Iterate descending so reductions that produce degree >= 12 are handled later in the loop.
    for t in (12..=22).rev() {
        let c = tmp[t];
        if c.is_zero() {
            continue;
        }
        tmp[t] = Fq::zero();

        // c * w^t = c * w^{t-12} * w^12 = c * w^{t-12} * (18 w^6 - 82)
        tmp[t - 6] += c18 * c;
        tmp[t - 12] -= c82 * c;
    }

    let mut out = [Fq::zero(); 12];
    out.copy_from_slice(&tmp[..12]);
    out
}

/// Packed witness for GT exponentiation
#[derive(Clone)]
pub struct GtExpWitness<F: JoltField> {
    /// rho(s, x) - all intermediate results packed into 11-var MLE
    pub rho_packed: Vec<F>,
    /// rho_next(s, x) = rho(s+1, x) - shifted intermediate results
    pub rho_next_packed: Vec<F>,
    /// quotient(s, x) - all quotients packed into 11-var MLE
    pub quotient_packed: Vec<F>,
    /// digit_lo(s) - low digit bit replicated across x
    pub digit_lo_packed: Vec<F>,
    /// digit_hi(s) - high digit bit replicated across x
    pub digit_hi_packed: Vec<F>,
    /// base(x) - base element replicated across s
    pub base_packed: Vec<F>,
    /// base2(x) - base^2 element replicated across s
    pub base2_packed: Vec<F>,
    /// base3(x) - base^3 element replicated across s
    pub base3_packed: Vec<F>,
    /// Number of actual steps
    pub num_steps: usize,
}

impl GtExpWitness<Fq> {
    /// Create packed witness from individual step data.
    ///
    /// Data layout: index = x * 128 + s (s in low 7 bits, x in high 4 bits).
    /// This allows LowToHigh binding to naturally give us:
    /// - Phase 1 (rounds 0-6): bind step variables s (low bits)
    /// - Phase 2 (rounds 7-10): bind element variables x (high bits)
    pub fn from_steps(
        rho_mles: &[Vec<Fq>],      // rho_mles[step][x] for step in 0..=num_steps
        quotient_mles: &[Vec<Fq>], // quotient_mles[step][x] for step in 0..num_steps
        bits: &[bool],             // bits (MSB first)
        base_mle: &[Fq],           // base[x] - 16 values
        base2_mle: &[Fq],          // base^2[x] - 16 values
        base3_mle: &[Fq],          // base^3[x] - 16 values
    ) -> Self {
        let mut digits = digits_from_bits_msb(bits);
        let num_steps = digits.len();
        assert_eq!(rho_mles.len(), num_steps + 1, "Need num_steps + 1 rho MLEs");
        assert_eq!(
            quotient_mles.len(),
            num_steps,
            "Need num_steps quotient MLEs"
        );
        assert_eq!(base_mle.len(), 16, "Base must be 4-var MLE (16 values)");
        assert_eq!(base2_mle.len(), 16, "Base2 must be 4-var MLE (16 values)");
        assert_eq!(base3_mle.len(), 16, "Base3 must be 4-var MLE (16 values)");

        let step_size = 1 << NUM_STEP_VARS; // 128
        let elem_size = 1 << NUM_ELEMENT_VARS; // 16
        let total_size = 1 << NUM_TOTAL_VARS; // 2048
        let num_steps_padded = (num_steps + 1).min(step_size);

        // If we have room, pad one extra "dummy" quotient row so that:
        // - rho_next is a pure shift of rho (terminal row shifts to zero), and
        // - the packed GT exp constraint is still satisfied on the Boolean hypercube.
        let mut quotient_mles_padded = quotient_mles.to_vec();
        if num_steps_padded > num_steps {
            let g_mle = get_g_mle();
            let rho_prev = &rho_mles[num_steps];

            let mut dummy_q = vec![Fq::zero(); elem_size];
            for j in 0..elem_size {
                let rho4 = rho_prev[j].square().square();
                if !g_mle[j].is_zero() {
                    dummy_q[j] = -rho4 / g_mle[j];
                }
            }

            quotient_mles_padded.push(dummy_q);
            digits.push((false, false));
        }

        // Pack rho: rho_packed[x * 128 + s] = rho_mles[s][x]
        // s in low 7 bits, x in high 4 bits
        let mut rho_packed = vec![Fq::zero(); total_size];
        // NOTE: Populate through num_steps_padded so rho_next is a pure shift of rho.
        for s in 0..num_steps_padded.min(step_size) {
            for x in 0..elem_size {
                if s < rho_mles.len() && x < rho_mles[s].len() {
                    rho_packed[x * step_size + s] = rho_mles[s][x];
                }
            }
        }

        // Pack rho_next: rho_next_packed[x * 128 + s] = rho_mles[s+1][x]
        let mut rho_next_packed = vec![Fq::zero(); total_size];
        for s in 0..num_steps_padded.min(step_size) {
            for x in 0..elem_size {
                if s + 1 < rho_mles.len() && x < rho_mles[s + 1].len() {
                    rho_next_packed[x * step_size + s] = rho_mles[s + 1][x];
                }
            }
        }

        // Pack quotient: quotient_packed[x * 128 + s] = quotient_mles[s][x]
        let mut quotient_packed = vec![Fq::zero(); total_size];
        for s in 0..num_steps_padded.min(step_size) {
            for x in 0..elem_size {
                if s < quotient_mles_padded.len() && x < quotient_mles_padded[s].len() {
                    quotient_packed[x * step_size + s] = quotient_mles_padded[s][x];
                }
            }
        }

        // Pack digit_lo and digit_hi: replicated across x
        let mut digit_lo_packed = vec![Fq::zero(); total_size];
        let mut digit_hi_packed = vec![Fq::zero(); total_size];
        for s in 0..num_steps_padded.min(step_size) {
            let (digit_hi, digit_lo) = digits[s];
            let lo_val = if digit_lo { Fq::from(1u64) } else { Fq::zero() };
            let hi_val = if digit_hi { Fq::from(1u64) } else { Fq::zero() };
            for x in 0..elem_size {
                digit_lo_packed[x * step_size + s] = lo_val;
                digit_hi_packed[x * step_size + s] = hi_val;
            }
        }

        // Pack base: base_packed[x * 128 + s] = base_mle[x] (replicated across s)
        // base only depends on x, so same value for all s with same x
        let mut base_packed = vec![Fq::zero(); total_size];
        for x in 0..elem_size {
            for s in 0..step_size {
                base_packed[x * step_size + s] = base_mle[x];
            }
        }

        // Pack base2: base2_packed[x * 128 + s] = base2_mle[x] (replicated across s)
        let mut base2_packed = vec![Fq::zero(); total_size];
        for x in 0..elem_size {
            for s in 0..step_size {
                base2_packed[x * step_size + s] = base2_mle[x];
            }
        }

        // Pack base3: base3_packed[x * 128 + s] = base3_mle[x] (replicated across s)
        let mut base3_packed = vec![Fq::zero(); total_size];
        for x in 0..elem_size {
            for s in 0..step_size {
                base3_packed[x * step_size + s] = base3_mle[x];
            }
        }

        Self {
            rho_packed,
            rho_next_packed,
            quotient_packed,
            digit_lo_packed,
            digit_hi_packed,
            base_packed,
            base2_packed,
            base3_packed,
            num_steps: num_steps_padded,
        }
    }
}

/// GT multiplication constraint polynomials
#[derive(Clone)]
pub struct GtMulConstraintPolynomials<F: JoltField> {
    pub lhs: Vec<F>,
    pub rhs: Vec<F>,
    pub result: Vec<F>,
    pub quotient: Vec<F>,
    pub constraint_index: usize,
}

/// Convert MSB-first scalar bits to base-4 digits (MSB-first).
pub fn digits_from_bits_msb(bits: &[bool]) -> Vec<(bool, bool)> {
    let mut padded = bits.to_vec();
    if padded.len() % 2 != 0 {
        padded.insert(0, false);
    }
    padded.chunks(2).map(|c| (c[0], c[1])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::recursion::curve::{Bn254Recursion, RecursionCurve};
    use ark_ff::UniformRand;

    #[test]
    fn poly12_mul_mod_g_matches_fq12_square_and_cube_under_coeff_map() {
        let mut rng = ark_std::test_rng();
        // Spot-check several random bases.
        for _ in 0..50 {
            let base = Fq12::rand(&mut rng);
            let base_coeffs = fq12_to_poly12_coeffs(&base);

            let sq_coeffs = poly12_mul_mod_g(&base_coeffs, &base_coeffs);
            let cube_coeffs = poly12_mul_mod_g(&sq_coeffs, &base_coeffs);

            let sq_native = base * base;
            let cube_native = sq_native * base;
            assert_eq!(sq_coeffs, fq12_to_poly12_coeffs(&sq_native));
            assert_eq!(cube_coeffs, fq12_to_poly12_coeffs(&cube_native));
        }
    }

    #[test]
    fn poly12_eval_with_power_sums_matches_materialized_mle_dot_product() {
        let mut rng = ark_std::test_rng();
        for _ in 0..20 {
            let base = Fq12::rand(&mut rng);
            let coeffs = fq12_to_poly12_coeffs(&base);
            // Random eq weights (not necessarily eq polynomial outputs, but the identity holds for any weights).
            let mut eq = vec![Fq::zero(); 16];
            for e in &mut eq {
                *e = Fq::rand(&mut rng);
            }
            let ps = poly12_power_sums_at_eq(&eq);
            let fast = poly12_eval_with_power_sums(&coeffs, &ps);

            let evals = <Bn254Recursion as RecursionCurve>::fq12_to_mle(&base);
            let slow: Fq = evals
                .iter()
                .zip(eq.iter())
                .map(|(v, w)| *v * *w)
                .fold(Fq::zero(), |acc, x| acc + x);

            assert_eq!(fast, slow);
        }
    }
}

// ============================================================================
// Eq polynomial utilities (LSB-first order)
// ============================================================================

/// Compute Eq evaluations in LSB-first order
pub fn eq_lsb_evals<F: JoltField>(r: &[F::Challenge]) -> Vec<F> {
    let n = r.len();
    let mut evals = vec![F::zero(); 1 << n];
    for idx in 0..(1 << n) {
        let mut prod = F::one();
        for i in 0..n {
            let bit = ((idx >> i) & 1) == 1;
            let r_i: F = r[i].into();
            let y_i = if bit { F::one() } else { F::zero() };
            prod *= r_i * y_i + (F::one() - r_i) * (F::one() - y_i);
        }
        evals[idx] = prod;
    }
    evals
}

/// Compute Eq MLE in LSB-first order
pub fn eq_lsb_mle<F: JoltField>(r: &[F::Challenge], y: &[F::Challenge]) -> F {
    let mut prod = F::one();
    for i in 0..r.len() {
        let r_i: F = r[i].into();
        let y_i: F = y[i].into();
        prod *= r_i * y_i + (F::one() - r_i) * (F::one() - y_i);
    }
    prod
}

/// Compute EqPlusOne MLE in LSB-first order
pub fn eq_plus_one_lsb_mle<F: JoltField>(r: &[F::Challenge], y: &[F::Challenge]) -> F {
    let n = r.len();
    let one = F::one();
    let mut sum = F::zero();
    for k in 0..n {
        let mut lower = F::one();
        for i in 0..k {
            let r_i: F = r[i].into();
            let y_i: F = y[i].into();
            lower *= r_i * (one - y_i);
        }

        let r_k: F = r[k].into();
        let y_k: F = y[k].into();
        let kth = (one - r_k) * y_k;

        let mut higher = F::one();
        for i in (k + 1)..n {
            let r_i: F = r[i].into();
            let y_i: F = y[i].into();
            higher *= r_i * y_i + (one - r_i) * (one - y_i);
        }

        sum += lower * kth * higher;
    }
    sum
}

/// Compute EqPlusOne evaluations in LSB-first order
pub fn eq_plus_one_lsb_evals<F: JoltField>(r: &[F::Challenge]) -> Vec<F> {
    let n = r.len();
    let mut evals = vec![F::zero(); 1 << n];
    for idx in 0..(1 << n) {
        let mut y = vec![F::Challenge::from(0u128); n];
        for i in 0..n {
            if ((idx >> i) & 1) == 1 {
                y[i] = F::Challenge::from(1u128);
            }
        }
        evals[idx] = eq_plus_one_lsb_mle::<F>(r, &y);
    }
    evals
}
