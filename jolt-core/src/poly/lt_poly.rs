use std::iter::zip;

use allocative::Allocative;

use crate::{
    field::JoltField,
    poly::{
        eq_poly::EqPolynomial,
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialBinding},
        opening_proof::{OpeningPoint, BIG_ENDIAN},
    },
};

#[derive(Allocative)]
pub struct LtPolynomial<F: JoltField> {
    lt_lo: MultilinearPolynomial<F>,
    lt_hi: MultilinearPolynomial<F>,
    eq_hi: MultilinearPolynomial<F>,
    n_lo_vars: usize,
}

impl<F: JoltField> LtPolynomial<F> {
    pub fn new(r_cycle: &OpeningPoint<BIG_ENDIAN, F>) -> Self {
        let (r_hi, r_lo) = r_cycle.split_at(r_cycle.len() / 2);
        Self {
            lt_lo: MultilinearPolynomial::from(lt_evals::<F>(&r_lo)),
            lt_hi: MultilinearPolynomial::from(lt_evals::<F>(&r_hi)),
            eq_hi: MultilinearPolynomial::from(EqPolynomial::<F>::evals(&r_hi.r)),
            n_lo_vars: r_lo.len(),
        }
    }

    pub fn bind(&mut self, r_j: F::Challenge, bind_order: BindingOrder) {
        match bind_order {
            BindingOrder::HighToLow => self.bind_high_to_low(r_j),
            BindingOrder::LowToHigh => self.bind_low_to_high(r_j),
        }
    }

    fn bind_high_to_low(&mut self, r_j: F::Challenge) {
        let n_hi_vars = self.lt_hi.get_num_vars();
        if n_hi_vars != 0 {
            self.lt_hi.bind_parallel(r_j, BindingOrder::HighToLow);
            self.eq_hi.bind_parallel(r_j, BindingOrder::HighToLow);
        } else {
            self.lt_lo.bind_parallel(r_j, BindingOrder::HighToLow);
            self.n_lo_vars -= 1;
        }
    }

    fn bind_low_to_high(&mut self, r_j: F::Challenge) {
        if self.n_lo_vars != 0 {
            self.lt_lo.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.n_lo_vars -= 1;
        } else {
            self.lt_hi.bind_parallel(r_j, BindingOrder::LowToHigh);
            self.eq_hi.bind_parallel(r_j, BindingOrder::LowToHigh);
        }
    }

    pub fn get_bound_coeff(&self, i: usize) -> F {
        let i_hi = i >> self.n_lo_vars;
        let i_lo = i & ((1 << self.n_lo_vars) - 1);
        // LT(i) = LT_hi(i_hi) + EQ_hi(i_hi) * LT_lo(i_lo)
        self.lt_hi.get_bound_coeff(i_hi)
            + self.eq_hi.get_bound_coeff(i_hi) * self.lt_lo.get_bound_coeff(i_lo)
    }
}

/// Returns the MLE of `LT(j, r)` evaluated at all Boolean `j ∈ {0,1}^n`.
///
/// The less-than MLE is defined as:
///   `LT(x, y) = Σ_i (1 - x_i) · y_i · eq(x[i+1:], y[i+1:])`
/// where the sum runs from MSB to LSB.
///
/// This function computes `[LT(0, r), LT(1, r), ..., LT(2^n - 1, r)]`.
pub fn lt_evals<F: JoltField>(r: &OpeningPoint<BIG_ENDIAN, F>) -> Vec<F> {
    let mut evals: Vec<F> = vec![F::zero(); 1 << r.len()];
    for (i, r) in r.r.iter().rev().enumerate() {
        let (evals_left, evals_right) = evals.split_at_mut(1 << i);
        zip(evals_left, evals_right).for_each(|(x, y)| {
            *y = *x * r;
            *x += *r - *y;
        });
    }
    evals
}

/// Returns LT suffix tables and EQ suffix tables for the given challenges.
///
/// The tables are built from LSB to MSB (processing r[n-1] first, then r[n-2], etc.),
/// which matches the index ordering used by [`lt_evals`].
///
/// - `lt_result[k]` = LT evals over the last k variables (r[n-k:])
/// - `eq_result[k]` = EQ evals over the last k variables (r[n-k:])
///
/// Index convention matches [`lt_evals`]:
/// - `lt_result[0] = [0]` (LT over 0 vars is always 0)
/// - `eq_result[0] = [1]` (EQ over 0 vars is always 1)
/// - `lt_result[k]` has `2^k` entries
///
/// The recurrence (processing from LSB r[n-1] to MSB r[0]):
/// For x with new MSB = 0: new_LT = old_LT * (1 - r_new) + r_new
/// For x with new MSB = 1: new_LT = old_LT * r_new
pub fn lt_evals_cached<F: JoltField>(
    r: &[F::Challenge],
) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
    let n = r.len();
    let mut lt_tables: Vec<Vec<F>> = Vec::with_capacity(n + 1);
    let mut eq_tables: Vec<Vec<F>> = Vec::with_capacity(n + 1);

    // Base case: 0 variables
    lt_tables.push(vec![F::zero()]); // LT over 0 vars is 0
    eq_tables.push(vec![F::one()]); // EQ over 0 vars is 1

    // Build tables from LSB (r[n-1]) to MSB (r[0]), matching lt_evals
    for k in 0..n {
        let r_k: F = r[n - 1 - k].into(); // Process from end to start
        let one_minus_r_k = F::one() - r_k;
        let prev_size = 1 << k;

        // Allocate new tables with doubled size
        let mut lt_curr = vec![F::zero(); prev_size * 2];
        let mut eq_curr = vec![F::zero(); prev_size * 2];

        let lt_prev = &lt_tables[k];
        let eq_prev = &eq_tables[k];

        // For each previous index, extend with new MSB bit
        for i in 0..prev_size {
            let lt_old = lt_prev[i];
            let eq_old = eq_prev[i];

            // Index i stays at position i (new MSB = 0)
            // new_LT = old_LT * (1 - r_k) + r_k
            lt_curr[i] = lt_old * one_minus_r_k + r_k;
            eq_curr[i] = eq_old * one_minus_r_k;

            // Index i + prev_size gets new MSB = 1
            // new_LT = old_LT * r_k
            lt_curr[i + prev_size] = lt_old * r_k;
            eq_curr[i + prev_size] = eq_old * r_k;
        }

        lt_tables.push(lt_curr);
        eq_tables.push(eq_curr);
    }

    (lt_tables, eq_tables)
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;

    use crate::{
        field::challenge::MontU128Challenge,
        poly::{
            multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialEvaluation},
            opening_proof::{OpeningPoint, BIG_ENDIAN},
        },
    };

    use super::{lt_evals, lt_evals_cached, LtPolynomial};

    #[test]
    fn test_bind_low_to_high_works() {
        let r_cycle = OpeningPoint::new([9, 5, 7, 1].map(MontU128Challenge::from).to_vec());
        let mut lt_poly = LtPolynomial::<Fr>::new(&r_cycle);
        let lt_poly_gt: MultilinearPolynomial<Fr> = lt_evals(&r_cycle).into();
        let r0 = MontU128Challenge::from(2);
        let r1 = MontU128Challenge::from(6);
        let r2 = MontU128Challenge::from(3);
        let r3 = MontU128Challenge::from(9);
        let r = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![r3, r2, r1, r0]);

        lt_poly.bind(r0, BindingOrder::LowToHigh);
        lt_poly.bind(r1, BindingOrder::LowToHigh);
        lt_poly.bind(r2, BindingOrder::LowToHigh);
        lt_poly.bind(r3, BindingOrder::LowToHigh);

        assert_eq!(lt_poly.get_bound_coeff(0), lt_poly_gt.evaluate(&r.r));
    }

    #[test]
    fn test_bind_high_to_low_works() {
        let r_cycle = OpeningPoint::new([9, 5, 7, 1].map(MontU128Challenge::from).to_vec());
        let mut lt_poly = LtPolynomial::<Fr>::new(&r_cycle);
        let lt_poly_gt: MultilinearPolynomial<Fr> = lt_evals(&r_cycle).into();
        let r0 = MontU128Challenge::from(2);
        let r1 = MontU128Challenge::from(6);
        let r2 = MontU128Challenge::from(3);
        let r3 = MontU128Challenge::from(9);
        let r = OpeningPoint::<BIG_ENDIAN, Fr>::new(vec![r0, r1, r2, r3]);

        lt_poly.bind(r0, BindingOrder::HighToLow);
        lt_poly.bind(r1, BindingOrder::HighToLow);
        lt_poly.bind(r2, BindingOrder::HighToLow);
        lt_poly.bind(r3, BindingOrder::HighToLow);

        assert_eq!(lt_poly.get_bound_coeff(0), lt_poly_gt.evaluate(&r.r));
    }

    #[test]
    fn test_lt_evals_cached() {
        use crate::poly::eq_poly::EqPolynomial;

        // Test that lt_evals_cached produces correct suffix tables
        // Tables are built from LSB to MSB, so lt_tables[k] represents
        // LT over the last k variables (r[n-k:])
        let r_cycle: OpeningPoint<BIG_ENDIAN, Fr> =
            OpeningPoint::new([9, 5, 7, 1].map(MontU128Challenge::from).to_vec());
        let (lt_tables, eq_tables) = lt_evals_cached::<Fr>(&r_cycle.r);
        let n = r_cycle.len();

        // Should have n+1 tables (0 to n variables)
        assert_eq!(lt_tables.len(), n + 1);
        assert_eq!(eq_tables.len(), n + 1);

        // Base case checks
        assert_eq!(lt_tables[0], vec![Fr::from(0u64)]);
        assert_eq!(eq_tables[0], vec![Fr::from(1u64)]);

        // Verify each suffix table matches the direct computation
        // lt_tables[k] = LT over r[n-k:]
        for k in 1..=n {
            let r_suffix = OpeningPoint::<BIG_ENDIAN, Fr>::new(r_cycle.r[n - k..].to_vec());
            let lt_direct = lt_evals::<Fr>(&r_suffix);
            let eq_direct = EqPolynomial::<Fr>::evals(&r_suffix.r);

            assert_eq!(
                lt_tables[k], lt_direct,
                "LT table mismatch at k={k}"
            );
            assert_eq!(
                eq_tables[k], eq_direct,
                "EQ table mismatch at k={k}"
            );
        }
    }

    #[test]
    fn test_lt_evals_cached_final_matches_lt_evals() {
        type Challenge = MontU128Challenge<Fr>;
        // Verify the final table of lt_evals_cached matches lt_evals
        for num_vars in 1..=8 {
            let r: Vec<Challenge> = (0..num_vars)
                .map(|i| Challenge::from((i * 7 + 3) as u128))
                .collect();
            let r_point = OpeningPoint::<BIG_ENDIAN, Fr>::new(r.clone());

            let (lt_tables, _) = lt_evals_cached::<Fr>(&r);
            let lt_direct = lt_evals::<Fr>(&r_point);

            assert_eq!(
                *lt_tables.last().unwrap(),
                lt_direct,
                "Final LT table mismatch for num_vars={num_vars}"
            );
        }
    }
}
