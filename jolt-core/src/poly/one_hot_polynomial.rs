use crate::field::JoltField;
use crate::msm::VariableBaseMSM;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::matrix_layout::MatrixLayout;
use crate::utils::math::Math;
use allocative::Allocative;
use ark_bn254::G1Affine;
use ark_ec::CurveGroup;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;

#[cfg(test)]
use crate::poly::dense_mlpoly::DensePolynomial;

/// Represents a one-hot multilinear polynomial (ra/wa) used
/// in Twist/Shout. Perhaps somewhat unintuitively, the implementation
/// in this file is currently only used to compute the Dory
/// commitment.
#[derive(Clone, Debug, Allocative)]
pub struct OneHotPolynomial<F: JoltField> {
    /// The size of the "address" space for this polynomial.
    pub K: usize,
    /// The indices of the nonzero coefficients for each j \in {0, 1}^T.
    /// In other words, the raf/waf corresponding to this
    /// ra/wa polynomial.
    /// If empty, this polynomial is 0 for all j.
    pub nonzero_indices: Arc<Vec<Option<u8>>>,
    _marker: PhantomData<F>,
}

impl<F: JoltField> PartialEq for OneHotPolynomial<F> {
    fn eq(&self, other: &Self) -> bool {
        self.K == other.K && self.nonzero_indices == other.nonzero_indices
    }
}

impl<F: JoltField> Default for OneHotPolynomial<F> {
    fn default() -> Self {
        Self {
            K: 1,
            nonzero_indices: Arc::new(vec![]),
            _marker: PhantomData,
        }
    }
}

impl<F: JoltField> OneHotPolynomial<F> {
    /// The number of rows in the coefficient matrix used to commit to this polynomial.
    pub fn num_rows(&self, layout: &MatrixLayout) -> usize {
        let t = self.nonzero_indices.len();
        layout.one_hot_num_rows(self.K, t)
    }

    pub fn get_num_vars(&self) -> usize {
        self.K.log_2() + self.nonzero_indices.len().log_2()
    }

    #[cfg(test)]
    fn to_dense_poly(&self, T: usize) -> DensePolynomial<F> {
        let mut dense_coeffs: Vec<F> = vec![F::zero(); self.K * T];
        for (t, k) in self.nonzero_indices.iter().enumerate() {
            if let Some(k) = k {
                let log_K = self.K.log_2();
                let k = (k & !((1 << log_K) - 1))
                    | ((k & ((1 << log_K) - 1)).reverse_bits() >> (u8::BITS as usize - log_K));
                dense_coeffs[k as usize * T + t] = F::one();
            }
        }
        DensePolynomial::new(dense_coeffs)
    }

    pub fn evaluate<C>(&self, r: &[C]) -> F
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        let log_t = self.nonzero_indices.len().log_2();
        let (r_address, r_cycle) = r.split_at(r.len() - log_t);
        let eq_r_address = EqPolynomial::<F>::evals(r_address);
        let eq_r_cycle = EqPolynomial::<F>::evals(r_cycle);

        self.nonzero_indices
            .par_iter()
            .zip(eq_r_cycle.par_iter())
            .map(|(k, eq_cycle)| match k {
                Some(k) => eq_r_address[*k as usize] * eq_cycle,
                None => F::zero(),
            })
            .sum()
    }

    pub fn from_indices(nonzero_indices: Vec<Option<u8>>, K: usize, T: usize) -> Self {
        debug_assert_eq!(T, nonzero_indices.len());
        assert!(K <= 1usize << u8::BITS, "K must be <= 256 for indices");

        Self {
            K,
            nonzero_indices: Arc::new(nonzero_indices),
            _marker: PhantomData,
        }
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomial::commit_rows")]
    pub fn commit_rows<G: CurveGroup<ScalarField = F> + VariableBaseMSM>(
        &self,
        bases: &[G::Affine],
        layout: &MatrixLayout,
    ) -> Vec<G> {
        let num_rows = self.num_rows(layout);
        let row_len = layout.num_columns;
        let t = self.nonzero_indices.len();

        debug_assert!(
            bases.len() >= row_len,
            "Expected at least row_len bases for row commitments"
        );

        // SAFETY: This function is only called with G1Affine
        let g1_bases = unsafe { std::mem::transmute::<&[G::Affine], &[G1Affine]>(bases) };

        let rows_per_k = t / row_len;
        if layout.cycle_major && rows_per_k >= rayon::current_num_threads() {
            let chunk_commitments: Vec<Vec<G>> = self
                .nonzero_indices
                .par_chunks(row_len)
                .map(|chunk| {
                    let mut indices_per_k: Vec<Vec<usize>> = vec![Vec::new(); self.K];
                    for (col_index, k) in chunk.iter().enumerate() {
                        if let Some(k) = k {
                            indices_per_k[*k as usize].push(col_index);
                        }
                    }
                    let results =
                        jolt_optimizations::batch_g1_additions_multi(g1_bases, &indices_per_k);
                    let mut row_commitments = vec![G::zero(); self.K];
                    for (k, batch_result) in results.into_iter().enumerate() {
                        if !indices_per_k[k].is_empty() {
                            let projective = ark_bn254::G1Projective::from(batch_result);
                            row_commitments[k] = unsafe { std::mem::transmute_copy(&projective) };
                        }
                    }
                    row_commitments
                })
                .collect();

            let mut result = vec![G::zero(); num_rows];
            for (chunk_index, commitments) in chunk_commitments.iter().enumerate() {
                result
                    .par_iter_mut()
                    .skip(chunk_index)
                    .step_by(rows_per_k)
                    .zip(commitments.par_iter())
                    .for_each(|(dest, src)| *dest = *src);
            }
            return result;
        }

        let mut row_indices: Vec<Vec<usize>> = vec![Vec::new(); num_rows];
        for (cycle, k) in self.nonzero_indices.iter().enumerate() {
            if let Some(k) = k {
                let global_index = layout.address_cycle_to_index(*k as usize, cycle, self.K, t);
                let row_index = global_index / row_len;
                let col_index = global_index % row_len;
                if row_index < num_rows {
                    row_indices[row_index].push(col_index);
                }
            }
        }

        let num_chunks = rayon::current_num_threads().next_power_of_two();
        let chunk_size = num_rows.div_ceil(num_chunks).max(1);
        let mut result: Vec<G> = vec![G::zero(); num_rows];

        result
            .par_chunks_mut(chunk_size)
            .zip(row_indices.par_chunks(chunk_size))
            .for_each(|(result_chunk, indices_chunk)| {
                let results = jolt_optimizations::batch_g1_additions_multi(g1_bases, indices_chunk);
                for (row_result, (indices, batch_result)) in result_chunk
                    .iter_mut()
                    .zip(indices_chunk.iter().zip(results.into_iter()))
                {
                    if !indices.is_empty() {
                        let projective = ark_bn254::G1Projective::from(batch_result);
                        *row_result = unsafe { std::mem::transmute_copy(&projective) };
                    }
                }
            });

        result
    }

    #[tracing::instrument(skip_all, name = "OneHotPolynomial::vector_matrix_product")]
    pub fn vector_matrix_product(
        &self,
        left_vec: &[F],
        coeff: F,
        result: &mut [F],
        layout: &MatrixLayout,
    ) {
        let t = self.nonzero_indices.len();
        let num_columns = layout.num_columns;
        debug_assert_eq!(result.len(), num_columns);

        if layout.cycle_major && t >= num_columns {
            let rows_per_k = t / num_columns;
            result
                .par_iter_mut()
                .enumerate()
                .for_each(|(col_index, dest)| {
                    let mut col_dot_product = F::zero();
                    for (row_offset, cycle) in (col_index..t).step_by(num_columns).enumerate() {
                        if let Some(k) = self.nonzero_indices[cycle] {
                            let row_index = k as usize * rows_per_k + row_offset;
                            col_dot_product += left_vec[row_index];
                        }
                    }
                    *dest += coeff * col_dot_product;
                });
            return;
        }

        for (cycle, k) in self.nonzero_indices.iter().enumerate() {
            if let Some(k) = k {
                let global_index = layout.address_cycle_to_index(*k as usize, cycle, self.K, t);
                let row_index = global_index / num_columns;
                let col_index = global_index % num_columns;
                if row_index < left_vec.len() && col_index < result.len() {
                    result[col_index] += coeff * left_vec[row_index];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::commitment::dory::{DoryContext, DoryGlobals, DoryLayout};
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_core::RngCore;
    use serial_test::serial;

    fn layout_from_globals() -> MatrixLayout {
        let (num_rows, num_columns) = DoryGlobals::matrix_shape();
        MatrixLayout {
            num_columns,
            num_rows,
            T: DoryGlobals::get_T(),
            cycle_major: DoryGlobals::get_layout() == DoryLayout::CycleMajor,
        }
    }

    fn evaluate_test<const LOG_K: usize, const LOG_T: usize>() {
        let K: usize = 1 << LOG_K;
        let T: usize = 1 << LOG_T;
        DoryGlobals::reset();
        let _guard = DoryGlobals::initialize_context(K, T, DoryContext::Main, None);
        let layout = layout_from_globals();

        let mut rng = test_rng();

        let nonzero_indices: Vec<_> = (0..T)
            .map(|_| Some((rng.next_u64() % K as u64) as u8))
            .collect();
        let one_hot_poly = OneHotPolynomial::<Fr>::from_indices(nonzero_indices, K, T);
        let dense_poly = one_hot_poly.to_dense_poly(layout.T);

        let mut r: Vec<<Fr as JoltField>::Challenge> =
            std::iter::repeat_with(|| <Fr as JoltField>::Challenge::random(&mut rng))
                .take(LOG_K + LOG_T)
                .collect();
        let r_one_hot = r.clone();
        r[..LOG_K].reverse();

        assert_eq!(one_hot_poly.evaluate(&r_one_hot), dense_poly.evaluate(&r));
    }

    #[test]
    #[serial]
    fn evaluate_K_less_than_T() {
        evaluate_test::<5, 6>();
    }

    #[test]
    #[serial]
    fn evaluate_K_equals_T() {
        evaluate_test::<6, 6>();
    }

    #[test]
    #[serial]
    fn evaluate_K_greater_than_T() {
        evaluate_test::<6, 5>();
    }
}
