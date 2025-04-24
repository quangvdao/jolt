use crate::field::JoltField;
use rayon::prelude::*;

use crate::utils::{math::Math, thread::unsafe_allocate_zero_vec};

pub struct EqPolynomial<F> {
    r: Vec<F>,
}

pub struct EqPlusOnePolynomial<F> {
    x: Vec<F>,
}

const PARALLEL_THRESHOLD: usize = 16;

impl<F: JoltField> EqPolynomial<F> {
    pub fn new(r: Vec<F>) -> Self {
        EqPolynomial { r }
    }

    pub fn evaluate(&self, rx: &[F]) -> F {
        assert_eq!(self.r.len(), rx.len());
        (0..rx.len())
            .map(|i| self.r[i] * rx[i] + (F::one() - self.r[i]) * (F::one() - rx[i]))
            .product()
    }

    #[tracing::instrument(skip_all, name = "EqPolynomial::evals")]
    /// Computes the table of coefficients: `{eq(r, x) for all x in {0, 1}^n}`
    pub fn evals(r: &[F]) -> Vec<F> {
        match r.len() {
            0..=PARALLEL_THRESHOLD => Self::evals_serial(r, None),
            _ => Self::evals_parallel(r, None),
        }
    }

    /// Extends an EQ evaluation table `eq(r_prefix, x)` to `eq((r_prefix, r_new), (x, y))`.
    ///
    /// Takes a slice `evals_prefix` representing evaluations of `eq(r_prefix, x)` for all `x` in `{0, 1}^k`,
    /// and a new challenge `r_new`. Returns a `Vec<F>` of length `2 * evals_prefix.len()` representing
    /// evaluations of `eq((r_prefix, r_new), z)` for all `z` in `{0, 1}^{k+1}`.
    pub fn extend_eq_evals(evals_prefix: &[F], r_new: F) -> Vec<F> {
        let prefix_len = evals_prefix.len();
        let new_len = prefix_len * 2;
        let mut new_evals = Vec::with_capacity(new_len);

        for i in 0..prefix_len {
            let scalar = evals_prefix[i];
            let eval_1 = scalar * r_new;      // Corresponds to y=1
            let eval_0 = scalar - eval_1;    // Corresponds to y=0
            new_evals.push(eval_0);
            new_evals.push(eval_1);
        }
        // The layout should match evals_serial: evals[2*i] is y=0, evals[2*i+1] is y=1
        // Check against evals_serial logic: (i is new index)
        // evals[i] = scalar * r[j];         -> new_evals[2*idx+1] = evals_prefix[idx] * r_new
        // evals[i-1] = scalar - evals[i]; -> new_evals[2*idx] = evals_prefix[idx] - new_evals[2*idx+1]
        // The current implementation matches this logic.

        new_evals
    }

    #[tracing::instrument(skip_all, name = "EqPolynomial::evals_cached")]
    /// Computes the table of coefficients like `evals`, but also caches the intermediate results
    ///
    /// In other words, computes `{eq(r[i..], x) for all x in {0, 1}^{n - i}}` and for all `i in
    /// 0..r.len()`.
    pub fn evals_cached(r: &[F]) -> Vec<Vec<F>> {
        // TODO: implement parallel version (it seems more difficult to parallelize all the
        // intermediate results)
        Self::evals_serial_cached(r, None)
    }

    /// When evaluating a multilinear polynomial on a point `r`, we first compute the EQ evaluation table
    /// for `r`, then dot-product those values with the coefficients of the polynomial.
    ///
    /// However, if the polynomial in question is a `CompactPolynomial`, its coefficients are represented
    /// by primitive integers while the dot product needs to be computed using Montgomery multiplication.
    ///
    /// To avoid converting every polynomial coefficient to Montgomery form, we can instead introduce an
    /// additional R^2 factor to every element in the EQ evaluation table and performing the dot product
    /// using `JoltField::mul_u64_unchecked`.
    ///
    /// We can efficiently compute the EQ table with this additional R^2 factor by initializing the root of
    /// the dynamic programming tree to R^2 instead of 1.
    #[tracing::instrument(skip_all, name = "EqPolynomial::evals_with_r2")]
    pub fn evals_with_r2(r: &[F]) -> Vec<F> {
        match r.len() {
            0..=PARALLEL_THRESHOLD => Self::evals_serial(r, F::montgomery_r2()),
            _ => Self::evals_parallel(r, F::montgomery_r2()),
        }
    }

    /// Computes the table of coefficients:
    ///     scaling_factor * eq(r, x) for all x in {0, 1}^n
    /// serially. More efficient for short `r`.
    fn evals_serial(r: &[F], scaling_factor: Option<F>) -> Vec<F> {
        let mut evals: Vec<F> = vec![scaling_factor.unwrap_or(F::one()); r.len().pow2()];
        let mut size = 1;
        for j in 0..r.len() {
            // in each iteration, we double the size of chis
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                // copy each element from the prior iteration twice
                let scalar = evals[i / 2];
                evals[i] = scalar * r[j];
                evals[i - 1] = scalar - evals[i];
            }
        }
        evals
    }

    /// Computes the table of coefficients like `evals_serial`, but also caches the intermediate results.
    ///
    /// Returns a vector of vectors, where the `j`th vector contains the coefficients for the polynomial
    /// `eq(r[..j], x)` for all `x in {0, 1}^{j}`.
    ///
    /// Performance seems at most 10% worse than `evals_serial`
    fn evals_serial_cached(r: &[F], scaling_factor: Option<F>) -> Vec<Vec<F>> {
        let mut evals: Vec<Vec<F>> = (0..r.len() + 1)
            .map(|i| vec![scaling_factor.unwrap_or(F::one()); 1 << i])
            .collect();
        let mut size = 1;
        for j in 0..r.len() {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let scalar = evals[j][i / 2];
                evals[j + 1][i] = scalar * r[j];
                evals[j + 1][i - 1] = scalar - evals[j + 1][i];
            }
        }
        evals
    }

    /// Computes the table of coefficients like `evals_serial`, but also caches the intermediate
    /// results. This binds `r` in the reverse order compared to `evals_serial_cached`.
    ///
    /// Concretely, this returns a vector of vectors, where the `(n -j)`th vector contains the
    /// coefficients for the polynomial `eq(r[j..], x)` for all `x in {0, 1}^{n - j}`.
    ///
    /// Performance seems at most 10% worse than `evals_serial`
    #[allow(dead_code)]
    fn evals_serial_cached_rev(r: &[F], scaling_factor: Option<F>) -> Vec<Vec<F>> {
        let rev_r = r.iter().rev().collect::<Vec<_>>();
        let mut evals: Vec<Vec<F>> = (0..r.len() + 1)
            .map(|i| vec![scaling_factor.unwrap_or(F::one()); 1 << i])
            .collect();
        let mut size = 1;
        for j in 0..r.len() {
            for i in 0..size {
                let scalar = evals[j][i];
                let multiple = 1 << j;
                evals[j + 1][i + multiple] = scalar * rev_r[j];
                evals[j + 1][i] = scalar - evals[j + 1][i + multiple];
            }
            size *= 2;
        }
        evals
    }

    /// Computes the table of coefficients:
    ///
    ///     `scaling_factor * eq(r, x) for all x in {0, 1}^n`,
    ///
    /// computing biggest layers of the dynamic programming tree in parallel.
    #[tracing::instrument(skip_all, "EqPolynomial::evals_parallel")]
    pub fn evals_parallel(r: &[F], scaling_factor: Option<F>) -> Vec<F> {
        let final_size = r.len().pow2();
        let mut evals: Vec<F> = unsafe_allocate_zero_vec(final_size);
        let mut size = 1;
        evals[0] = scaling_factor.unwrap_or(F::one());

        for r in r.iter().rev() {
            let (evals_left, evals_right) = evals.split_at_mut(size);
            let (evals_right, _) = evals_right.split_at_mut(size);

            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * *r;
                    *x -= *y;
                });

            size *= 2;
        }

        evals
    }
}

impl<F: JoltField> EqPlusOnePolynomial<F> {
    pub fn new(x: Vec<F>) -> Self {
        EqPlusOnePolynomial { x }
    }

    /* This MLE is 1 if y = x + 1 for x in the range [0... 2^l-2].
    That is, it ignores the case where x is all 1s, outputting 0.
    Assumes x and y are provided big-endian. */
    pub fn evaluate(&self, y: &[F]) -> F {
        let l = self.x.len();
        let x = &self.x;
        assert!(y.len() == l);
        let one = F::from_u64(1_u64);

        /* If y+1 = x, then the two bit vectors are of the following form.
            Let k be the longest suffix of 1s in x.
            In y, those k bits are 0.
            Then, the next bit in x is 0 and the next bit in y is 1.
            The remaining higher bits are the same in x and y.
        */
        (0..l)
            .into_par_iter()
            .map(|k| {
                let lower_bits_product = (0..k)
                    .map(|i| x[l - 1 - i] * (F::one() - y[l - 1 - i]))
                    .product::<F>();
                let kth_bit_product = (F::one() - x[l - 1 - k]) * y[l - 1 - k];
                let higher_bits_product = ((k + 1)..l)
                    .map(|i| {
                        x[l - 1 - i] * y[l - 1 - i] + (one - x[l - 1 - i]) * (one - y[l - 1 - i])
                    })
                    .product::<F>();
                lower_bits_product * kth_bit_product * higher_bits_product
            })
            .sum()
    }

    #[tracing::instrument(skip_all, "EqPlusOnePolynomial::evals")]
    pub fn evals(r: &[F], scaling_factor: Option<F>) -> (Vec<F>, Vec<F>) {
        let ell = r.len();
        let mut eq_evals: Vec<F> = unsafe_allocate_zero_vec(ell.pow2());
        eq_evals[0] = scaling_factor.unwrap_or(F::one());
        let mut eq_plus_one_evals: Vec<F> = unsafe_allocate_zero_vec(ell.pow2());

        // i indicates the LENGTH of the prefix of r for which the eq_table is calculated
        let eq_evals_helper = |eq_evals: &mut Vec<F>, r: &[F], i: usize| {
            debug_assert!(i != 0);
            let step = 1 << (ell - i); // step = (full / size)/2

            let mut selected: Vec<_> = eq_evals.par_iter_mut().step_by(step).collect();

            selected.par_chunks_mut(2).for_each(|chunk| {
                *chunk[1] = *chunk[0] * r[i - 1];
                *chunk[0] -= *chunk[1];
            });
        };

        for i in 0..ell {
            let step = 1 << (ell - i);
            let half_step = step / 2;

            let r_lower_product = (F::one() - r[i]) * r.iter().skip(i + 1).copied().product::<F>();

            eq_plus_one_evals
                .par_iter_mut()
                .enumerate()
                .skip(half_step)
                .step_by(step)
                .for_each(|(index, v)| {
                    *v = eq_evals[index - half_step] * r_lower_product;
                });

            eq_evals_helper(&mut eq_evals, r, i + 1);
        }

        (eq_evals, eq_plus_one_evals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use std::time::Instant;

    #[test]
    /// Test that the results of running `evals_serial`, `evals_parallel`, and `evals_serial_cached`
    /// (taking the last vector) are the same (and also benchmark them)
    fn test_evals() {
        let mut rng = test_rng();
        for len in 5..22 {
            let r = (0..len).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
            let start = Instant::now();
            let evals_serial = EqPolynomial::evals_serial(&r, None);
            let end_first = Instant::now();
            let evals_parallel = EqPolynomial::evals_parallel(&r, None);
            let end_second = Instant::now();
            let evals_serial_cached = EqPolynomial::evals_serial_cached(&r, None);
            let end_third = Instant::now();
            println!(
                "len: {}, Time taken to compute evals_serial: {:?}",
                len,
                end_first - start
            );
            println!(
                "len: {}, Time taken to compute evals_parallel: {:?}",
                len,
                end_second - end_first
            );
            println!(
                "len: {}, Time taken to compute evals_serial_cached: {:?}",
                len,
                end_third - end_second
            );
            assert_eq!(evals_serial, evals_parallel);
            assert_eq!(evals_serial, *evals_serial_cached.last().unwrap());
        }
    }

    #[test]
    /// Test that the `i`th vector of `evals_serial_cached` is equivalent to
    /// `evals(&r[..i])`, for all `i`.
    fn test_evals_cached() {
        let mut rng = test_rng();
        for len in 2..22 {
            let r = (0..len).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
            let evals_serial_cached = EqPolynomial::evals_serial_cached(&r, None);
            for i in 0..len {
                let evals = EqPolynomial::evals(&r[..i]);
                assert_eq!(evals_serial_cached[i], evals);
            }
        }
    }

    #[test]
    fn test_extend_eq_evals() {
        let mut rng = test_rng();

        // Goal: Verify that `extend_eq_evals` correctly performs one step of the iterative
        // EQ table computation performed by `evals`.
        // Strategy: Compare the output of `extend_eq_evals(evals(r_prefix), r_new)`
        // against the direct computation `evals([r_prefix, r_new].concat())`.

        // Loop through prefix lengths 0 to 9
        for len in 0..10 {
            // --- Setup ---
            // Generate random challenges for the prefix and the new element.
            let r_prefix: Vec<Fr> = (0..len).map(|_| Fr::random(&mut rng)).collect();
            let r_new = Fr::random(&mut rng);

            // --- Step 1: Calculate EQ table for the prefix challenges --- 
            // evals_prefix = [eq(r_prefix, x) for all x in {0,1}^len]
            // If len = 0, r_prefix = [], evals_prefix = [1].
            let evals_prefix = EqPolynomial::evals(&r_prefix);
            assert_eq!(evals_prefix.len(), 1 << len);

            // --- Step 2: Use the new function to extend the table with r_new --- 
            // extended_evals should be [eq((r_prefix, r_new), z) for z in {0,1}^{len+1}]
            let extended_evals = EqPolynomial::extend_eq_evals(&evals_prefix, r_new);
            assert_eq!(extended_evals.len(), 1 << (len + 1));

            // --- Step 3: Calculate EQ table for the full challenge vector directly (ground truth) --- 
            let r_full = [r_prefix.as_slice(), &[r_new]].concat(); // Combine prefix and new challenge
            // expected_evals = [eq((r_prefix, r_new), z) for z in {0,1}^{len+1}]
            let expected_evals = EqPolynomial::evals(&r_full);
            assert_eq!(expected_evals.len(), 1 << (len + 1));

            // --- Step 4: Compare results from Step 2 and Step 3 --- 
            // They should be identical because extend_eq_evals performs exactly one step of the evals logic.
            assert_eq!(extended_evals, expected_evals, "Failed for prefix length {}", len);
        }
    }
}
