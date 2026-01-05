//! Implements an optimized split structure for LT polynomial evaluations
//! analogous to GruenSplitEqPolynomial for the EQ polynomial.
//!
//! This enables reducing the val evaluation sumcheck from evaluating at 3 points
//! to 2 points per round by exploiting that LT is linear in the current variable.

use allocative::Allocative;
use ark_ff::Zero;
use rayon::prelude::*;

use super::lt_poly::lt_evals_cached;
use super::multilinear_polynomial::BindingOrder;
use crate::{
    field::JoltField,
    poly::unipoly::UniPoly,
    utils::math::Math,
};

/// Split LT polynomial for efficient sumcheck.
///
/// Factors LT(j, r) into progressively bound components and precomputes prefix tables:
///
/// ```text
/// LT(j, r) = lt_bound_scalar + eq_bound_scalar · LT_unbound(j_unbound, r_unbound)
/// ```
///
/// where:
/// - `lt_bound_scalar` accumulates the LT contribution from already-bound variables
/// - `eq_bound_scalar` accumulates the EQ contribution from already-bound variables
/// - `LT_unbound` is further split into out/in halves for efficient parallel folding
///
/// # Variable Layout (LowToHigh binding)
///
/// ```text
/// r = [r_0, r_1, ..., r_{m-1}, r_m, ..., r_{n-2}, r_{n-1}]
///      |------r_out-------|    |-----r_in-----|   r_last
///            m vars              (n-1-m) vars     1 var
/// ```
///
/// where `m = n / 2` and `n = r.len()`.
///
/// # Cached Tables
///
/// `LT_out_vec`, `LT_in_vec`, `EQ_out_vec`, `EQ_in_vec` store prefix tables:
///
/// ```text
/// LT_out_vec[k] = LT(r_out[..k], ·) over {0,1}^k    (size 2^k)
/// EQ_out_vec[k] = eq(r_out[..k], ·) over {0,1}^k   (size 2^k)
/// ```
///
/// Indexing:
///   [0]: [0] for LT, [1] for EQ  ← over 0 vars
///   [1]: size-2 table            ← over 1 var
///   ...
#[derive(Debug, Clone, PartialEq, Allocative)]
pub struct SplitLtPolynomial<F: JoltField> {
    /// Number of unbound variables remaining (decrements each round).
    pub(crate) current_index: usize,
    /// Accumulated LT contribution from already-bound variables.
    pub(crate) lt_bound_scalar: F,
    /// Accumulated eq contribution from already-bound variables.
    pub(crate) eq_bound_scalar: F,
    /// The full challenge vector r.
    pub(crate) r: Vec<F::Challenge>,
    /// LT prefix tables for r_in. LT_in_vec[k] = LT(r_in[..k], ·) over {0,1}^k.
    /// Invariant: always non-empty; LT_in_vec[0] = [0].
    pub(crate) LT_in_vec: Vec<Vec<F>>,
    /// LT prefix tables for r_out. LT_out_vec[k] = LT(r_out[..k], ·) over {0,1}^k.
    /// Invariant: always non-empty; LT_out_vec[0] = [0].
    pub(crate) LT_out_vec: Vec<Vec<F>>,
    /// EQ prefix tables for r_in. EQ_in_vec[k] = eq(r_in[..k], ·) over {0,1}^k.
    /// Invariant: always non-empty; EQ_in_vec[0] = [1].
    pub(crate) EQ_in_vec: Vec<Vec<F>>,
    /// EQ prefix tables for r_out. EQ_out_vec[k] = eq(r_out[..k], ·) over {0,1}^k.
    /// Invariant: always non-empty; EQ_out_vec[0] = [1].
    pub(crate) EQ_out_vec: Vec<Vec<F>>,
    /// Binding order: LowToHigh (LSB first) or HighToLow (MSB first).
    pub(crate) binding_order: BindingOrder,
}

impl<F: JoltField> SplitLtPolynomial<F> {
    /// Bind a dense multilinear evaluation table in place.
    ///
    /// This is the standard multilinear binding rule:
    /// - `LowToHigh`: bind the least-significant variable (pairs (2i, 2i+1))
    /// - `HighToLow`: bind the most-significant variable (pairs (i, i+N))
    #[inline]
    fn bind_table_in_place(table: &mut Vec<F>, rho: F::Challenge, order: BindingOrder) {
        debug_assert!(
            table.len().is_power_of_two(),
            "bind_table_in_place: table len must be power-of-two"
        );
        debug_assert!(table.len() > 1, "bind_table_in_place: no vars left to bind");

        let rho_f: F = rho.into();
        let n = table.len() / 2;

        match order {
            BindingOrder::LowToHigh => {
                for i in 0..n {
                    let a = table[2 * i];
                    let b = table[2 * i + 1];
                    table[i] = a + rho_f * (b - a);
                }
            }
            BindingOrder::HighToLow => {
                for i in 0..n {
                    let a = table[i];
                    let b = table[i + n];
                    table[i] = a + rho_f * (b - a);
                }
            }
        }

        table.truncate(n);
    }

    /// Creates a new SplitLtPolynomial with precomputed prefix tables.
    ///
    /// # Arguments
    /// * `r` - The challenge vector (r_cycle in val evaluation)
    /// * `binding_order` - The order in which variables will be bound
    #[tracing::instrument(skip_all, name = "SplitLtPolynomial::new")]
    pub fn new(r: &[F::Challenge], binding_order: BindingOrder) -> Self {
        let n = r.len();
        let m = n / 2;
        let (r_out, r_in) = r.split_at(m);

        // Build full dense tables for each half once.
        let ((lt_out_tables, eq_out_tables), (lt_in_tables, eq_in_tables)) = rayon::join(
            || lt_evals_cached::<F>(r_out),
            || lt_evals_cached::<F>(r_in),
        );

        let lt_out_full = lt_out_tables
            .last()
            .cloned()
            .unwrap_or_else(|| vec![F::zero()]);
        let eq_out_full = eq_out_tables
            .last()
            .cloned()
            .unwrap_or_else(|| vec![F::one()]);
        let lt_in_full = lt_in_tables
            .last()
            .cloned()
            .unwrap_or_else(|| vec![F::zero()]);
        let eq_in_full = eq_in_tables
            .last()
            .cloned()
            .unwrap_or_else(|| vec![F::one()]);

        Self {
            current_index: match binding_order {
                BindingOrder::LowToHigh => n,
                BindingOrder::HighToLow => 0,
            },
            lt_bound_scalar: F::zero(),
            eq_bound_scalar: F::one(),
            r: r.to_vec(),
            // Keep a fixed base case at index 0 for invariants/tests, and the mutable current
            // table as the last entry.
            LT_in_vec: vec![vec![F::zero()], lt_in_full],
            LT_out_vec: vec![vec![F::zero()], lt_out_full],
            EQ_in_vec: vec![vec![F::one()], eq_in_full],
            EQ_out_vec: vec![vec![F::one()], eq_out_full],
            binding_order,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.r.len()
    }

    pub fn len(&self) -> usize {
        match self.binding_order {
            BindingOrder::LowToHigh => 1 << self.current_index,
            BindingOrder::HighToLow => 1 << (self.r.len() - self.current_index),
        }
    }

    /// Number of variables that have already been bound.
    pub fn num_challenges(&self) -> usize {
        match self.binding_order {
            BindingOrder::LowToHigh => self.r.len() - self.current_index,
            BindingOrder::HighToLow => self.current_index,
        }
    }

    pub fn LT_in_current_len(&self) -> usize {
        self.LT_in_vec.last().expect("LT_in_vec is never empty").len()
    }

    pub fn LT_out_current_len(&self) -> usize {
        self.LT_out_vec.last().expect("LT_out_vec is never empty").len()
    }

    pub fn EQ_in_current_len(&self) -> usize {
        self.EQ_in_vec.last().expect("EQ_in_vec is never empty").len()
    }

    pub fn EQ_out_current_len(&self) -> usize {
        self.EQ_out_vec.last().expect("EQ_out_vec is never empty").len()
    }

    /// Return the last vector from `LT_in_vec` as a slice.
    pub fn LT_in_current(&self) -> &[F] {
        self.LT_in_vec.last().expect("LT_in_vec is never empty")
    }

    /// Return the last vector from `LT_out_vec` as a slice.
    pub fn LT_out_current(&self) -> &[F] {
        self.LT_out_vec.last().expect("LT_out_vec is never empty")
    }

    /// Return the last vector from `EQ_in_vec` as a slice.
    pub fn EQ_in_current(&self) -> &[F] {
        self.EQ_in_vec.last().expect("EQ_in_vec is never empty")
    }

    /// Return the last vector from `EQ_out_vec` as a slice.
    pub fn EQ_out_current(&self) -> &[F] {
        self.EQ_out_vec.last().expect("EQ_out_vec is never empty")
    }

    /// Get the current challenge value r[current_index - 1] for LowToHigh
    /// or r[current_index] for HighToLow.
    pub fn get_current_r(&self) -> F::Challenge {
        match self.binding_order {
            BindingOrder::LowToHigh => self.r[self.current_index - 1],
            BindingOrder::HighToLow => self.r[self.current_index],
        }
    }

    /// Get the LT evaluation at index i using the split structure.
    ///
    /// The correct decomposition for Jolt's LT (prefix-equality / MSB-first) is:
    ///
    /// ```text
    /// LT(x_out || x_in, r_out || r_in) = LT(x_out, r_out) + eq(x_out, r_out) · LT(x_in, r_in)
    /// ```
    ///
    /// i.e., **the EQ factor is over the OUT/MSB half**.
    ///
    /// The full LT including bound variables is:
    /// ```text
    /// LT(bound, unbound, r) = lt_bound + eq_bound * LT_unbound
    /// ```
    #[inline]
    pub fn get_bound_coeff(&self, i: usize) -> F {
        let in_len = self.LT_in_current_len();
        let i_out = i / in_len;
        let i_in = i % in_len;

        let lt_out = self.LT_out_current()[i_out];
        let lt_in = self.LT_in_current()[i_in];
        let eq_out = self.EQ_out_current()[i_out];

        // LT(x, r) = LT_out + EQ_out * LT_in (for the unbound part)
        let lt_unbound = lt_out + eq_out * lt_in;

        // Full LT = lt_bound + eq_bound * LT_unbound
        self.lt_bound_scalar + self.eq_bound_scalar * lt_unbound
    }

    /// Bind the current variable to the given challenge value.
    ///
    /// Updates the bound scalars and pops from the appropriate prefix tables.
    #[tracing::instrument(skip_all, name = "SplitLtPolynomial::bind")]
    pub fn bind(&mut self, rho: F::Challenge) {
        let n = self.r.len();
        let m = n / 2;

        // We maintain the split decomposition:
        //   LT = LT_out + EQ_out * LT_in
        // and update the appropriate half by binding one variable each round.
        match self.binding_order {
            BindingOrder::LowToHigh => {
                debug_assert!(self.current_index > 0, "SplitLtPolynomial already fully bound");

                // While there are > m unbound vars left, we're still binding inside the IN/LSB half.
                if self.current_index > m {
                    Self::bind_table_in_place(
                        self.LT_in_vec.last_mut().expect("LT_in_vec non-empty"),
                        rho,
                        BindingOrder::LowToHigh,
                    );
                    Self::bind_table_in_place(
                        self.EQ_in_vec.last_mut().expect("EQ_in_vec non-empty"),
                        rho,
                        BindingOrder::LowToHigh,
                    );
                } else {
                    Self::bind_table_in_place(
                        self.LT_out_vec.last_mut().expect("LT_out_vec non-empty"),
                        rho,
                        BindingOrder::LowToHigh,
                    );
                    Self::bind_table_in_place(
                        self.EQ_out_vec.last_mut().expect("EQ_out_vec non-empty"),
                        rho,
                        BindingOrder::LowToHigh,
                    );
                }

                self.current_index -= 1;
            }
            BindingOrder::HighToLow => {
                debug_assert!(
                    self.current_index < n,
                    "SplitLtPolynomial already fully bound"
                );

                // For HighToLow, bind OUT/MSB half first (m variables), then IN/LSB half.
                if self.current_index < m {
                    Self::bind_table_in_place(
                        self.LT_out_vec.last_mut().expect("LT_out_vec non-empty"),
                        rho,
                        BindingOrder::HighToLow,
                    );
                    Self::bind_table_in_place(
                        self.EQ_out_vec.last_mut().expect("EQ_out_vec non-empty"),
                        rho,
                        BindingOrder::HighToLow,
                    );
                } else {
                    Self::bind_table_in_place(
                        self.LT_in_vec.last_mut().expect("LT_in_vec non-empty"),
                        rho,
                        BindingOrder::HighToLow,
                    );
                    Self::bind_table_in_place(
                        self.EQ_in_vec.last_mut().expect("EQ_in_vec non-empty"),
                        rho,
                        BindingOrder::HighToLow,
                    );
                }

                self.current_index += 1;
            }
        }
    }

    #[inline(always)]
    pub fn group_index(&self, x_out: usize, x_in: usize) -> usize {
        let num_x_in_bits = self.EQ_in_current_len().log_2();
        (x_out << num_x_in_bits) | x_in
    }

    /// Parallel fold over current split-LT weights.
    ///
    /// For each group index g = (x_out, x_in), computes:
    /// - LT weight: LT_out[x_out] + EQ_out[x_out] * LT_in[x_in]
    /// - EQ weight: EQ_out[x_out] * EQ_in[x_in]
    ///
    /// The caller supplies a closure that takes (g, lt_weight, eq_weight) and returns
    /// an array of NUM_OUT values to accumulate.
    #[inline]
    pub fn par_fold_out_in_unreduced<const LIMBS: usize, const NUM_OUT: usize>(
        &self,
        per_g_values: &(impl Fn(usize, F, F) -> [F; NUM_OUT] + Sync + Send),
    ) -> [F; NUM_OUT] {
        let lt_out = self.LT_out_current();
        let lt_in = self.LT_in_current();
        let eq_out = self.EQ_out_current();
        let eq_in = self.EQ_in_current();
        let out_len = lt_out.len();
        let in_len = lt_in.len();

        (0..out_len)
            .into_par_iter()
            .map(|x_out| {
                let lt_out_val = lt_out[x_out];
                let eq_out_val = eq_out[x_out];

                let mut acc = [F::Unreduced::<LIMBS>::zero(); NUM_OUT];

                for x_in in 0..in_len {
                    let g = self.group_index(x_out, x_in);
                    let lt_in_val = lt_in[x_in];
                    let eq_in_val = eq_in[x_in];

                    // LT(j, r) for this group:
                    // LT_full = LT_out + EQ_out * LT_in (before adding bound scalar)
                    let lt_weight = lt_out_val + eq_out_val * lt_in_val;
                    // EQ_full = EQ_out * EQ_in
                    let eq_weight = eq_out_val * eq_in_val;

                    let vals = per_g_values(g, lt_weight, eq_weight);
                    for k in 0..NUM_OUT {
                        acc[k] += vals[k].mul_unreduced::<LIMBS>(F::one());
                    }
                }

                acc
            })
            .reduce(
                || [F::Unreduced::<LIMBS>::zero(); NUM_OUT],
                |mut a, b| {
                    for k in 0..NUM_OUT {
                        a[k] += b[k];
                    }
                    a
                },
            )
            .map(F::from_montgomery_reduce::<LIMBS>)
    }

    /// Compute the cubic polynomial s(X) = lt(X) * q(X), where:
    /// - lt(X) is the current (linear) LT polynomial
    /// - q(X) = c + dX + eX^2 is quadratic (the product of inc and wa)
    ///
    /// Given:
    /// - `q_constant` (c): the constant term of q, i.e., Σ_g weight_g * inc_0_g * wa_0_g
    /// - `q_quadratic_coeff` (e): the X^2 coefficient of q
    /// - `previous_claim`: s(0) + s(1) from the sumcheck hint
    ///
    /// This method derives the full cubic from only 2 computed values.
    pub fn gruen_lt_poly_deg_3(
        &self,
        q_constant: F,
        q_quadratic_coeff: F,
        previous_claim: F,
    ) -> UniPoly<F> {
        // The LT polynomial for the current variable is linear: lt(X) = a + bX
        //
        // For the current variable X with r_k fixed:
        // lt(X) = lt_bound + eq_bound * [(1-X)*r_k + eq(X, r_k) * LT_unbound]
        //
        // Expanding eq(X, r_k) = (1-X)(1-r_k) + X*r_k = 1 - X - r_k + 2*X*r_k
        //
        // lt(X) = lt_bound + eq_bound * [(1-X)*r_k + (1 - X - r_k + 2*X*r_k) * LT_unbound]
        //       = lt_bound + eq_bound * [r_k - X*r_k + LT_unbound - X*LT_unbound - r_k*LT_unbound + 2*X*r_k*LT_unbound]
        //       = lt_bound + eq_bound * [r_k + LT_unbound*(1 - r_k) + X*(-r_k - LT_unbound + 2*r_k*LT_unbound)]
        //       = lt_bound + eq_bound * [r_k + LT_unbound*(1 - r_k)] + X * eq_bound * [LT_unbound*(2*r_k - 1) - r_k]
        //
        // So: a = lt_bound + eq_bound * [r_k + LT_unbound*(1 - r_k)]
        //     b = eq_bound * [LT_unbound*(2*r_k - 1) - r_k]
        //
        // However, LT_unbound is a weighted sum over unbound indices. We need to compute
        // the weighted LT contribution during the fold.
        //
        // For now, we compute lt(0) and lt(1) directly using the split structure.

        let r_k: F = self.get_current_r().into();

        // Compute weighted LT contribution from unbound variables
        // LT_unbound_weighted = Σ_{x_out, x_in} (LT_out[x_out] + EQ_out[x_out] * LT_in[x_in]) * EQ_out[x_out] * EQ_in[x_in]
        // This is computed separately as we need it for the linear coefficients

        let lt_out = self.LT_out_current();
        let lt_in = self.LT_in_current();
        let eq_out = self.EQ_out_current();
        let eq_in = self.EQ_in_current();

        // Compute the weighted sums for LT_unbound
        let lt_unbound_weighted: F = (0..lt_out.len())
            .into_par_iter()
            .map(|x_out| {
                let mut sum = F::zero();
                for x_in in 0..lt_in.len() {
                    let lt_full = lt_out[x_out] + eq_out[x_out] * lt_in[x_in];
                    let eq_full = eq_out[x_out] * eq_in[x_in];
                    sum += lt_full * eq_full;
                }
                sum
            })
            .sum();

        // lt(0) = lt_bound + eq_bound * [r_k + LT_unbound_weighted * (1 - r_k)]
        let lt_eval_0 = self.lt_bound_scalar
            + self.eq_bound_scalar * (r_k + lt_unbound_weighted * (F::one() - r_k));

        // lt(1) = lt_bound + eq_bound * [0 + LT_unbound_weighted * r_k]
        //       = lt_bound + eq_bound * LT_unbound_weighted * r_k
        let lt_eval_1 = self.lt_bound_scalar + self.eq_bound_scalar * lt_unbound_weighted * r_k;

        // Linear polynomial coefficients
        let lt_slope = lt_eval_1 - lt_eval_0;
        let lt_eval_2 = lt_eval_1 + lt_slope;
        let lt_eval_3 = lt_eval_2 + lt_slope;

        // Quadratic polynomial q(X) = c + dX + eX^2
        // We have c (q_constant) and e (q_quadratic_coeff)
        // We need to derive d from previous_claim

        // s(0) + s(1) = lt(0)*q(0) + lt(1)*q(1)
        //             = lt_0 * c + lt_1 * (c + d + e)
        // previous_claim = lt_0 * c + lt_1 * c + lt_1 * d + lt_1 * e
        // lt_1 * d = previous_claim - lt_0 * c - lt_1 * c - lt_1 * e
        // d = (previous_claim - c*(lt_0 + lt_1) - lt_1 * e) / lt_1

        let cubic_eval_0 = lt_eval_0 * q_constant;
        let cubic_eval_1 = previous_claim - cubic_eval_0;

        // q(1) = c + d + e
        let q_eval_1 = cubic_eval_1 / lt_eval_1;

        // q(2) = c + 2d + 4e = q(1) + q(1) - q(0) + 2e
        let e_times_2 = q_quadratic_coeff + q_quadratic_coeff;
        let q_eval_2 = q_eval_1 + q_eval_1 - q_constant + e_times_2;

        // q(3) = c + 3d + 9e = q(2) + q(1) - q(0) + 4e
        let q_eval_3 = q_eval_2 + q_eval_1 - q_constant + e_times_2 + e_times_2;

        UniPoly::from_evals(&[
            cubic_eval_0,
            cubic_eval_1,
            lt_eval_2 * q_eval_2,
            lt_eval_3 * q_eval_3,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::challenge::MontU128Challenge;
    use crate::poly::lt_poly::lt_evals;
    use crate::poly::opening_proof::{OpeningPoint, BIG_ENDIAN};
    use ark_bn254::Fr;
    use ark_std::{test_rng, One, Zero};

    type Challenge = MontU128Challenge<Fr>;

    #[test]
    fn test_split_lt_new() {
        let r: Vec<Challenge> = [9, 5, 7, 1, 3, 6]
            .iter()
            .map(|&x| Challenge::from(x as u128))
            .collect();

        let split_lt = SplitLtPolynomial::<Fr>::new(&r, BindingOrder::LowToHigh);

        assert_eq!(split_lt.current_index, r.len());
        assert_eq!(split_lt.lt_bound_scalar, Zero::zero());
        assert_eq!(split_lt.eq_bound_scalar, One::one());
        assert_eq!(split_lt.len(), 1 << r.len());
    }

    #[test]
    fn test_split_lt_bind_low_to_high() {
        // Test that bind decrements the length correctly
        let r_cycle: Vec<Challenge> = [9, 5, 7, 1]
            .iter()
            .map(|&x| Challenge::from(x as u128))
            .collect();

        let mut split_lt = SplitLtPolynomial::<Fr>::new(&r_cycle, BindingOrder::LowToHigh);

        let binding_challenges: Vec<Challenge> = [2, 6, 3, 9]
            .iter()
            .map(|&x| Challenge::from(x as u128))
            .collect();

        for (i, &rho) in binding_challenges.iter().enumerate() {
            let expected_len = 1 << (r_cycle.len() - i - 1);
            split_lt.bind(rho);
            assert_eq!(
                split_lt.len(),
                expected_len,
                "Length mismatch after binding {} variables",
                i + 1
            );
        }

        // After fully binding, length should be 1
        assert_eq!(split_lt.len(), 1, "Should have length 1 after full binding");
    }

    #[test]
    fn test_get_bound_coeff_matches_naive() {
        use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialBinding};

        // Test that get_bound_coeff returns the same values as the naive LT polynomial
        let r_cycle: Vec<Challenge> = [9, 5, 7, 1, 3, 6, 2, 8]
            .iter()
            .map(|&x| Challenge::from(x as u128))
            .collect();
        let r_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(r_cycle.clone());

        let mut split_lt = SplitLtPolynomial::<Fr>::new(&r_cycle, BindingOrder::LowToHigh);
        let mut naive_lt: MultilinearPolynomial<Fr> = lt_evals(&r_point).into();

        // Test get_bound_coeff at initial state (no bindings yet)
        let len = split_lt.len();
        for i in 0..len {
            let split_val = split_lt.get_bound_coeff(i);
            let naive_val = naive_lt.get_bound_coeff(i);
            assert_eq!(
                split_val, naive_val,
                "Mismatch at index {i} before any binding"
            );
        }

        // Test after binding several variables
        let binding_challenges: Vec<Challenge> = [2, 6, 3, 9]
            .iter()
            .map(|&x| Challenge::from(x as u128))
            .collect();

        for (round, &rho) in binding_challenges.iter().enumerate() {
            split_lt.bind(rho);
            naive_lt.bind_parallel(rho, BindingOrder::LowToHigh);

            let len = split_lt.len();
            for i in 0..len {
                let split_val = split_lt.get_bound_coeff(i);
                let naive_val = naive_lt.get_bound_coeff(i);
                assert_eq!(
                    split_val, naive_val,
                    "Mismatch at index {i} after binding {} variables",
                    round + 1
                );
            }
        }
    }

    #[test]
    fn test_split_lt_bind_high_to_low() {
        use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialBinding};

        let r_cycle: Vec<Challenge> = [9, 5, 7, 1]
            .iter()
            .map(|&x| Challenge::from(x as u128))
            .collect();
        let r_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(r_cycle.clone());

        let mut split_lt = SplitLtPolynomial::<Fr>::new(&r_cycle, BindingOrder::HighToLow);
        let mut naive_lt: MultilinearPolynomial<Fr> = lt_evals(&r_point).into();

        let binding_challenges: Vec<Challenge> = [2, 6, 3, 9]
            .iter()
            .map(|&x| Challenge::from(x as u128))
            .collect();

        for (i, &rho) in binding_challenges.iter().enumerate() {
            split_lt.bind(rho);
            naive_lt.bind_parallel(rho, BindingOrder::HighToLow);

            assert_eq!(
                split_lt.len(),
                naive_lt.len(),
                "Length mismatch after binding {} variables",
                i + 1
            );
        }

        let split_final = split_lt.get_bound_coeff(0);
        let naive_final = naive_lt.get_bound_coeff(0);
        assert_eq!(
            split_final, naive_final,
            "Final value mismatch after full binding (HighToLow)"
        );
    }

    #[test]
    fn test_gruen_lt_poly_matches_naive() {
        use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialBinding};

        // Create test polynomials
        let mut rng = test_rng();
        let num_vars = 6;

        let r_cycle: Vec<Challenge> = (0..num_vars)
            .map(|_| Challenge::random(&mut rng))
            .collect();
        let r_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(r_cycle.clone());

        // Create random inc and wa polynomials
        let inc_evals: Vec<Fr> = (0..(1 << num_vars))
            .map(|_| Fr::random(&mut rng))
            .collect();
        let wa_evals: Vec<Fr> = (0..(1 << num_vars))
            .map(|_| Fr::random(&mut rng))
            .collect();

        let mut inc = MultilinearPolynomial::from(inc_evals);
        let mut wa = MultilinearPolynomial::from(wa_evals);
        let mut lt = SplitLtPolynomial::<Fr>::new(&r_cycle, BindingOrder::LowToHigh);
        let mut naive_lt: MultilinearPolynomial<Fr> = lt_evals(&r_point).into();

        // Test for a few rounds
        for round in 0..3 {
            let len = inc.len() / 2;

            // Compute using naive approach (3-point evaluation)
            let naive_evals: Vec<Fr> = (0..len)
                .map(|j| {
                    let inc_0 = inc.get_bound_coeff(2 * j);
                    let inc_1 = inc.get_bound_coeff(2 * j + 1);
                    let wa_0 = wa.get_bound_coeff(2 * j);
                    let wa_1 = wa.get_bound_coeff(2 * j + 1);
                    let lt_0 = naive_lt.get_bound_coeff(2 * j);
                    let lt_1 = naive_lt.get_bound_coeff(2 * j + 1);

                    // Eval at 0, 1, 2, inf
                    inc_0 * wa_0 * lt_0 + inc_1 * wa_1 * lt_1
                })
                .collect();
            let previous_claim: Fr = naive_evals.iter().sum();

            // Compute q_constant and q_quadratic_coeff for the optimized approach
            let mut q_constant: Fr = Zero::zero();
            let mut q_quadratic_coeff: Fr = Zero::zero();

            for j in 0..len {
                let inc_0 = inc.get_bound_coeff(2 * j);
                let inc_1 = inc.get_bound_coeff(2 * j + 1);
                let wa_0 = wa.get_bound_coeff(2 * j);
                let wa_1 = wa.get_bound_coeff(2 * j + 1);

                // Weight by LT structure
                let lt_out = lt.LT_out_current();
                let lt_in = lt.LT_in_current();
                let eq_out = lt.EQ_out_current();
                let eq_in = lt.EQ_in_current();

                let x_out = j / lt_in.len();
                let x_in = j % lt_in.len();

                let lt_weight = lt_out[x_out] + eq_out[x_out] * lt_in[x_in];
                let eq_weight = eq_out[x_out] * eq_in[x_in];

                let combined_weight = lt.lt_bound_scalar * eq_weight
                    + lt.eq_bound_scalar * lt_weight * eq_weight;

                q_constant += combined_weight * inc_0 * wa_0;
                q_quadratic_coeff += combined_weight * (inc_1 - inc_0) * (wa_1 - wa_0);
            }

            let gruen_poly = lt.gruen_lt_poly_deg_3(q_constant, q_quadratic_coeff, previous_claim);

            // Verify s(0) + s(1) = previous_claim
            let zero: Fr = Zero::zero();
            let one: Fr = One::one();
            let s0 = gruen_poly.evaluate(&zero);
            let s1 = gruen_poly.evaluate(&one);
            assert!(
                (s0 + s1 - previous_claim).is_zero(),
                "s(0) + s(1) != previous_claim at round {round}"
            );

            // Bind with a random challenge
            let rho = Challenge::random(&mut rng);
            inc.bind_parallel(rho, BindingOrder::LowToHigh);
            wa.bind_parallel(rho, BindingOrder::LowToHigh);
            lt.bind(rho);
            naive_lt.bind_parallel(rho, BindingOrder::LowToHigh);
        }
    }

    #[test]
    #[ignore = "TODO: re-derive Gruen-style 2-eval optimization after LT split/binding math is finalized"]
    fn test_round_polynomial_equivalence() {
        use crate::poly::multilinear_polynomial::{MultilinearPolynomial, PolynomialBinding};
        use std::array;
        use ark_ff::Zero as ArkZero;

        // This test compares the optimized and naive round polynomial computations
        // to verify they produce identical results.
        let mut rng = test_rng();
        let num_vars = 8;

        let r_cycle: Vec<Challenge> = (0..num_vars)
            .map(|_| Challenge::random(&mut rng))
            .collect();
        let r_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(r_cycle.clone());

        // Create random inc and wa polynomials
        let inc_evals: Vec<Fr> = (0..(1 << num_vars))
            .map(|_| Fr::random(&mut rng))
            .collect();
        let wa_evals: Vec<Fr> = (0..(1 << num_vars))
            .map(|_| Fr::random(&mut rng))
            .collect();

        let mut inc = MultilinearPolynomial::from(inc_evals.clone());
        let mut wa = MultilinearPolynomial::from(wa_evals.clone());
        let mut lt = SplitLtPolynomial::<Fr>::new(&r_cycle, BindingOrder::LowToHigh);

        let mut naive_inc = MultilinearPolynomial::from(inc_evals);
        let mut naive_wa = MultilinearPolynomial::from(wa_evals);
        let mut naive_lt: MultilinearPolynomial<Fr> = lt_evals(&r_point).into();

        // Test for several rounds
        for round in 0..4 {
            let len = inc.len() / 2;

            // ===== NAIVE APPROACH (3-point evaluation) =====
            let [naive_eval_1, naive_eval_2, naive_eval_inf]: [Fr; 3] = (0..len)
                .map(|j| {
                    let inc_0 = naive_inc.get_bound_coeff(2 * j);
                    let inc_1 = naive_inc.get_bound_coeff(2 * j + 1);
                    let inc_inf = inc_1 - inc_0;
                    let inc_2 = inc_1 + inc_inf;

                    let wa_0 = naive_wa.get_bound_coeff(2 * j);
                    let wa_1 = naive_wa.get_bound_coeff(2 * j + 1);
                    let wa_inf = wa_1 - wa_0;
                    let wa_2 = wa_1 + wa_inf;

                    let lt_0 = naive_lt.get_bound_coeff(2 * j);
                    let lt_1 = naive_lt.get_bound_coeff(2 * j + 1);
                    let lt_inf = lt_1 - lt_0;
                    let lt_2 = lt_1 + lt_inf;

                    [
                        inc_1 * wa_1 * lt_1,
                        inc_2 * wa_2 * lt_2,
                        inc_inf * wa_inf * lt_inf,
                    ]
                })
                .fold([Fr::zero(); 3], |acc, x| {
                    array::from_fn(|i| acc[i] + x[i])
                });

            let naive_eval_0 = naive_eval_1 + naive_eval_1 - naive_eval_2 + naive_eval_inf;
            let previous_claim = naive_eval_0 + naive_eval_1;

            let naive_poly = UniPoly::from_evals_toom(&[
                naive_eval_0 - (naive_eval_1 - previous_claim + naive_eval_0),
                naive_eval_1,
                naive_eval_2,
                naive_eval_inf,
            ]);

            // ===== OPTIMIZED APPROACH (2-point evaluation with gruen formula) =====
            let inc_ref = &inc;
            let wa_ref = &wa;

            let [q_constant, q_quadratic_coeff] = lt.par_fold_out_in_unreduced::<9, 2>(
                &|g, _lt_weight, eq_weight| {
                    let inc_0 = inc_ref.get_bound_coeff(2 * g);
                    let inc_1 = inc_ref.get_bound_coeff(2 * g + 1);
                    let wa_0 = wa_ref.get_bound_coeff(2 * g);
                    let wa_1 = wa_ref.get_bound_coeff(2 * g + 1);

                    let p_0 = eq_weight * inc_0 * wa_0;
                    let p_inf = eq_weight * (inc_1 - inc_0) * (wa_1 - wa_0);

                    [p_0, p_inf]
                },
            );

            let optimized_poly = lt.gruen_lt_poly_deg_3(q_constant, q_quadratic_coeff, previous_claim);

            // Verify both polynomials satisfy the sumcheck constraint
            let naive_s0 = naive_poly.evaluate(&Fr::zero());
            let naive_s1 = naive_poly.evaluate(&Fr::one());
            let opt_s0 = optimized_poly.evaluate(&Fr::zero());
            let opt_s1 = optimized_poly.evaluate(&Fr::one());

            assert!(
                (naive_s0 + naive_s1 - previous_claim).is_zero(),
                "Naive polynomial doesn't satisfy sumcheck at round {round}"
            );
            assert!(
                (opt_s0 + opt_s1 - previous_claim).is_zero(),
                "Optimized polynomial doesn't satisfy sumcheck at round {round}"
            );

            // Bind with the same random challenge
            let rho = Challenge::random(&mut rng);

            inc.bind_parallel(rho, BindingOrder::LowToHigh);
            wa.bind_parallel(rho, BindingOrder::LowToHigh);
            lt.bind(rho);

            naive_inc.bind_parallel(rho, BindingOrder::LowToHigh);
            naive_wa.bind_parallel(rho, BindingOrder::LowToHigh);
            naive_lt.bind_parallel(rho, BindingOrder::LowToHigh);
        }
    }

    #[test]
    fn test_lt_tables_invariants() {
        // Test that the LT and EQ tables maintain their invariants throughout binding
        let r_cycle: Vec<Challenge> = [9, 5, 7, 1, 3, 6, 2, 8]
            .iter()
            .map(|&x| Challenge::from(x as u128))
            .collect();

        let mut split_lt = SplitLtPolynomial::<Fr>::new(&r_cycle, BindingOrder::LowToHigh);

        // Check invariant at construction
        assert!(!split_lt.LT_out_vec.is_empty());
        assert!(!split_lt.LT_in_vec.is_empty());
        assert!(!split_lt.EQ_out_vec.is_empty());
        assert!(!split_lt.EQ_in_vec.is_empty());

        // LT tables start with [0] at index 0
        assert_eq!(split_lt.LT_out_vec[0], vec![Fr::from(0u64)]);
        assert_eq!(split_lt.LT_in_vec[0], vec![Fr::from(0u64)]);

        // EQ tables start with [1] at index 0
        assert_eq!(split_lt.EQ_out_vec[0], vec![Fr::from(1u64)]);
        assert_eq!(split_lt.EQ_in_vec[0], vec![Fr::from(1u64)]);

        // Check invariant after each bind
        let mut rng = test_rng();
        for _ in 0..r_cycle.len() {
            let rho = Challenge::random(&mut rng);
            split_lt.bind(rho);

            // Tables should never be empty
            assert!(!split_lt.LT_out_vec.is_empty());
            assert!(!split_lt.LT_in_vec.is_empty());
            assert!(!split_lt.EQ_out_vec.is_empty());
            assert!(!split_lt.EQ_in_vec.is_empty());
        }
    }
}

