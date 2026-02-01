//! Shared types for GT group operations.

use crate::field::JoltField;
use crate::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};
use crate::zkvm::recursion::constraints::config::CONFIG;
use ark_bn254::Fq;
use ark_ff::Zero;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_optimizations::get_g_mle;

/// Number of step variables (7 for 128 base-4 steps)
pub const NUM_STEP_VARS: usize = CONFIG.step_vars;

/// Number of element variables (4 for Fq12 = 12 field elements, but we use 16 for power of 2)
pub const NUM_ELEMENT_VARS: usize = CONFIG.element_vars;

/// Total variables = step + element
pub const NUM_TOTAL_VARS: usize = CONFIG.packed_vars;

/// Public inputs for a single packed GT exponentiation.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct GtExpPublicInputs {
    /// Scalar bits (MSB-first, no leading zeros)
    pub scalar_bits: Vec<bool>,
}

impl GuestSerialize for GtExpPublicInputs {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.scalar_bits.guest_serialize(w)?;
        Ok(())
    }
}

impl GuestDeserialize for GtExpPublicInputs {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            scalar_bits: Vec::<bool>::guest_deserialize(r)?,
        })
    }
}

impl GtExpPublicInputs {
    /// Create new public inputs for a GT exponentiation
    pub fn new(scalar_bits: Vec<bool>) -> Self {
        Self { scalar_bits }
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
        let digits_len = n.div_ceil(2);
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
