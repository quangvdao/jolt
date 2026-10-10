//! Allocation-free pairwise XOR merge over disjoint mutable field slices.

use jolt_field::F128;
use rayon::prelude::*;

pub fn tree_merge<T: Send>(
    arrays: &mut [T],
    chunk_len: usize,
    as_slice: impl for<'a> Fn(&'a mut T) -> &'a mut [F128] + Copy + Sync,
) {
    let mut stride = 1;
    while stride < arrays.len() {
        arrays.par_chunks_mut(stride * 2).for_each(|pair| {
            if pair.len() > stride {
                let (left, right) = pair.split_at_mut(stride);
                as_slice(&mut left[0])
                    .par_chunks_mut(chunk_len)
                    .zip(as_slice(&mut right[0]).par_chunks(chunk_len))
                    .for_each(|(left, right)| {
                        for (left, &right) in left.iter_mut().zip(right) {
                            *left += right;
                        }
                    });
            }
        });
        stride *= 2;
    }
}
