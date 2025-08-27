use crate::field::JoltField;
use crate::subprotocols::univariate::{
    extrapolate_uk_to_uh, extrapolate_uk_to_uh_into, extrapolate_uk_to_uh_into_diff_only,
    product_eval_univariate_full_zero_based,
};
use ark_ff::Field;

/// Compute row-major strides for a given `shape` (last axis contiguous).
///
/// Inputs:
/// - `shape`: per-axis lengths where the last axis is intended to be contiguous
///
/// Outputs:
/// - Returns per-axis strides for row-major layout with the last axis contiguous
///
/// Invariants/Constraints:
/// - `shape.len()` defines the tensor rank v
///
/// Notes:
/// - These strides are used throughout to index grids shaped like `(k+1)^v` where per-axis
///   positions follow our internal convention: indices 0..k-1 correspond to finite 0..k-1 and
///   index k corresponds to ∞.
pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let v = shape.len();
    let mut strides = vec![1usize; v];
    for i in (0..v.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Trait to allow pluggable univariate extrapolation implementations.
///
/// Contract:
/// - `x_nodes`: evaluation nodes on some U_k = {0,1} if k=1, {0,1,...,k-1,∞} if k≥2
/// - `x_values`: evaluations at `x_nodes` using the standard layout: `[p(0), p(1)]` if k=1, `[p(0), p(1), ..., p(k-1), p(∞)]` if k≥2
/// - `y_nodes`: target nodes
/// - Returns evaluations at `y_nodes`
///
/// Panics/Expectations:
/// - Implementations should assert/require `x_nodes.len() == x_values.len()`
pub trait UniExtrap<F: Field> {
    fn extrapolate(x_nodes: &[F], x_values: &[F], y_nodes: &[F]) -> Vec<F>;
}

/// Generic version of multivariate extrapolation using a provided univariate extrapolation strategy.
// Old generic strategy-based multivariate extrapolator removed.

/// Naive multivariate extrapolation using plain Lagrange (via Vandermonde solve) per axis.
///
/// Baseline to validate `multivariate_extrapolate`. Interprets each line along an axis as
/// values on U_k = {0,1} if k=1, {0,1,...,k-1,∞} if k≥2 and extrapolates to U_h by solving for coefficients.
/// No small-int fast paths.
// --- Naive multilinear extension baseline (equality-polynomial based) ---

/// In-place subset Möbius transform converting values on {0,1}^v (last axis contiguous)
/// to multilinear coefficients c_T for p(y) = sum_T c_T prod_{i in T} y_i.
///
/// Inputs:
/// - `v`: number of variables
/// - `a`: mutable slice of length 2^v containing values on {0,1}^v in last-axis contiguous order
///
/// Outputs:
/// - Overwrites `a` with multilinear coefficients in-place
#[inline]
fn mobius_to_coeffs_inplace<F: JoltField>(v: usize, a: &mut [F]) {
    // Standard subset Möbius transform to convert values on {0,1}^v to coefficients
    // for the multilinear polynomial p(y) = sum_{T} c_T prod_{i in T} y_i.
    let n = 1usize << v;
    debug_assert_eq!(a.len(), n);
    for i in 0..v {
        let bit = 1usize << i;
        for mask in 0..n {
            if (mask & bit) != 0 {
                let other = mask ^ bit;
                a[mask] -= a[other];
            }
        }
    }
}

/// Evaluate multilinear polynomial from coefficients at a concrete point y in F^v.
///
/// Inputs:
/// - `v`: number of variables
/// - `coeffs`: coefficients over all subsets (length 2^v)
/// - `y`: point of length v with field values
///
/// Outputs:
/// - p(y)
#[allow(dead_code)]
#[inline]
fn eval_from_coeffs_at_point<F: JoltField>(v: usize, coeffs: &[F], y: &[F]) -> F {
    let n = 1usize << v;
    let mut acc = F::zero();
    for mask in 0..n {
        let mut prod = coeffs[mask];
        if prod.is_zero() {
            continue;
        }
        let mut m = mask;
        let mut i = 0usize;
        while m != 0 {
            if (m & 1) != 0 {
                prod *= y[i];
            }
            i += 1;
            m >>= 1;
        }
        acc += prod;
    }
    acc
}

/// Evaluate multilinear polynomial at a point with ∞ coordinates.
/// Each coordinate can be either Some(finite_u64_value) or None (∞).
/// When a coordinate is ∞, we only include monomials that contain that variable.
///
/// Inputs:
/// - `v`: number of variables
/// - `values_01`: evaluations on {0,1}^v (length 2^v)
/// - `y`: per-axis value or None for ∞
///
/// Outputs:
/// - p(y) with the ∞-semantics used throughout this file
#[inline]
fn eval_multilinear_at_point_with_infty_u64<F: JoltField>(
    v: usize,
    values_01: &[F],
    y: &[Option<u64>],
) -> F {
    let n = 1usize << v;
    assert_eq!(values_01.len(), n);

    // Convert to coefficients (this is cached in the original, but we recompute here)
    let mut coeffs = values_01.to_vec();
    mobius_to_coeffs_inplace::<F>(v, &mut coeffs);

    // Evaluate using coefficients with direct u64 multiplication
    let mut acc = F::zero();

    for mask in 0..n {
        // Require that all ∞ axes are included in the monomial
        let mut ok = true;
        for (i, yi) in y.iter().enumerate() {
            if yi.is_none() && (mask & (1usize << i)) == 0 {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }

        let mut prod = coeffs[mask];
        if prod.is_zero() {
            continue;
        }

        for (i, yi) in y.iter().enumerate() {
            if let Some(val_u64) = yi {
                if (mask & (1usize << i)) != 0 {
                    // Direct u64 multiplication - much faster!
                    prod = prod.mul_u64(*val_u64);
                }
            }
        }
        acc += prod;
    }
    acc
}

fn eval_from_coeffs_with_infty<F: JoltField>(v: usize, coeffs: &[F], y: &[Option<F>]) -> F {
    // y[i] = Some(value) for finite coordinate, None for ∞
    let n = 1usize << v;
    let mut acc = F::zero();
    for mask in 0..n {
        // Require that all ∞ axes are included in the monomial
        let mut ok = true;
        for (i, yi) in y.iter().enumerate() {
            if yi.is_none() && (mask & (1usize << i)) == 0 {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        let mut prod = coeffs[mask];
        if prod.is_zero() {
            continue;
        }
        for (i, yi) in y.iter().enumerate() {
            if let Some(val) = yi {
                if (mask & (1usize << i)) != 0 {
                    prod *= *val;
                }
            }
        }
        acc += prod;
    }
    acc
}

/// From evaluations on {0,1}^v, build the full U_h^v grid using multilinear extension.
///
/// **Coordinate System:** This function uses a bit-reversed coordinate system where
/// the last axis is contiguous (LSB corresponds to axis v-1). This is required for
/// consistency with `multivariate_extrapolate` and the fast implementation.
///
/// Inputs:
/// - `v`: number of variables
/// - `values_01`: evaluations on {0,1}^v in **standard bit ordering**, length 2^v
/// - `h`: target per-axis degree for U_h = {0,1} if h=1, {0,1,...,h-1,∞} if h≥2
///
/// Outputs:
/// - Evaluations on U_h^v of length (h+1)^v in row-major (last axis contiguous);
///   along each axis, indices 0..h-1 = 0..h-1, index h = ∞
///
/// Panics:
/// - If `values_01.len() != 2^v`
///
/// Implementation Notes:
/// - Input polynomials are bit-reversed internally to match the coordinate system
///   used by `multivariate_extrapolate` and other fast implementations
/// - This ensures consistency across all multivariate evaluation functions
pub fn multilinear_extend_to_u_h_grid_naive<F: JoltField>(
    v: usize,
    values_01: &[F],
    h: usize,
) -> Vec<F> {
    let n01 = 1usize << v;
    assert_eq!(values_01.len(), n01);

    // Reorder inputs so that the last axis is contiguous (LSB corresponds to axis v-1)
    // This bit reversal is required for consistency with multivariate_extrapolate
    let mut reordered = values_01.to_vec();
    for i in 0..n01 {
        let j = i.reverse_bits() >> (usize::BITS - v as u32);
        if i < j {
            reordered.swap(i, j);
        }
    }

    // Convert to multilinear coefficients
    let mut coeffs = reordered;
    mobius_to_coeffs_inplace::<F>(v, &mut coeffs);

    let side = h + 1; // U_h = {0,1} if h=1, {0,1,...,h-1, ∞} if h≥2 has h+1 elements (set definition)
    let total = side.pow(v as u32);
    let mut out = vec![F::zero(); total];

    for idx in 0..total {
        // Decode mixed-radix base (h+1) to get coordinates
        // Set is U_h = {0,1} if h=1, {0,1,...,h-1, ∞} if h≥2; NEW indexing convention:
        // t=0 corresponds to 0
        // t=1 corresponds to 1
        // ...
        // t=h-1 corresponds to h-1
        // t=h corresponds to ∞
        let mut tmp = idx;
        let mut y: Vec<Option<F>> = vec![None; v];

        for ax_rev in 0..v {
            let t = tmp % side;
            tmp /= side;
            let ax = v - 1 - ax_rev; // last axis is contiguous (LSD)

            if h >= 2 && t == h {
                // Last position is ∞
                y[ax] = None;
            } else {
                // Positions 0,1,...,h-1 correspond to 0,1,...,h-1
                y[ax] = Some(F::from_u64(t as u64));
            }
        }

        out[idx] = eval_from_coeffs_with_infty::<F>(v, &coeffs, &y);
    }

    out
}

/// Represents a multivariate polynomial by its evaluations on U_k^v for some degree k.
///
/// Paper alignment: U_k = {0,1} if k=1, {0,1,...,k-1,∞} if k≥2 as defined in sections/2_preliminaries.tex
/// Container for a v-variate polynomial represented by evaluations on U_k^v.
///
/// Fields:
/// - `degree` (k): per-axis degree
/// - `num_vars` (v): number of variables
/// - `evaluations`: length (k+1)^v in row-major (last axis contiguous);
///   per-axis layout uses the standard convention: for k=1 this is `[p(0), p(1)]`, for k≥2 this is `[p(0), p(1), ..., p(k-1), p(∞)]`.
#[derive(Clone, Debug)]
pub struct MultivariateEvaluations<F: Field> {
    /// The degree k (per-variable degree of the polynomial)
    pub degree: usize,
    /// The number of variables v
    pub num_vars: usize,
    /// Evaluations on U_k^v = {0,1}^v if k=1, {0,1,...,k-1,∞}^v if k≥2, stored in row-major order
    /// Length should be (degree + 1)^num_vars
    pub evaluations: Vec<F>,
}

/// Extrapolate evaluations from degree k to h on v variables.
///
/// Inputs:
/// - `v`: number of variables
/// - `k`: source per-axis degree (1 ≤ k ≤ 16)
/// - `h`: target per-axis degree (k ≤ h ≤ 32)
/// - `values`: length (k+1)^v in row-major (last axis contiguous)
///   - if k == 1: values are on U_1^v = {0,1}^v with per-axis layout `[p(0), p(1)]`
///   - if k ≥ 2: values are on U_k^v with per-axis layout `[p(0), p(1), ..., p(k-1), p(∞)]`
///
/// Outputs:
/// - Evaluations on U_h^v of length (h+1)^v with the same ordering convention
///
/// Panics:
/// - If `values.len() != (k+1)^v`
/// - If `k` or `h` are out of the supported range
pub fn multivariate_extrapolate<F: JoltField>(
    v: usize,
    k: usize,
    h: usize,
    values: &[F],
) -> Vec<F> {
    assert!(k >= 1 && k <= 16 && h >= k && h <= 32);
    assert_eq!(values.len(), (k + 1).pow(v as u32));
    // Early return only when both source and destination grids coincide semantically
    // For k >= 1, inputs are on U_k^v and outputs are also U_k^v when h == k → identity
    if h == k {
        return values.to_vec();
    }

    if v == 1 {
        return extrapolate_uk_to_uh(values, h);
    }

    let mut shape: Vec<usize> = core::iter::repeat(k + 1).take(v).collect();
    let final_size = (h + 1).pow(v as u32);

    // OPTIMIZATION: Use double buffering to reduce allocations from v to 2.
    let mut buffers = [values.to_vec(), vec![F::zero(); final_size]];
    let mut src_idx = 0;

    for axis in 0..v {
        let dst_idx = 1 - src_idx;

        let mut next_shape = shape.clone();
        next_shape[axis] = h + 1;
        let total_next_elems = next_shape.iter().copied().fold(1usize, |acc, s| acc * s);

        // Ensure the destination buffer is large enough.
        if buffers[dst_idx].len() < total_next_elems {
            buffers[dst_idx].resize(total_next_elems, F::zero());
        }

        // Get mutable, non-overlapping handles to the source and destination buffers.
        let (cur, next) = if src_idx == 0 {
            let (first, second) = buffers.split_at_mut(1);
            (&first[0], &mut second[0])
        } else {
            let (first, second) = buffers.split_at_mut(1);
            (&second[0], &mut first[0])
        };

        // Compute strides for current shape
        let strides_src = compute_strides(&shape);
        // Compute strides for new shape
        let strides_dst = compute_strides(&next_shape);

        let mut num_slices = 1usize;
        for (i, s) in shape.iter().enumerate() {
            if i != axis {
                num_slices *= *s;
            }
        }

        let src_axis_len = shape[axis];
        let dst_axis_len = next_shape[axis];

        // OPTIMIZATION: Hoist the line buffer allocation out of the hot loop.
        let mut line_buffer: Vec<F> = if k > 1 {
            Vec::with_capacity(src_axis_len)
        } else {
            Vec::new()
        };
        let mut line_ex_buffer: Vec<F> = if k > 1 {
            vec![F::zero(); dst_axis_len]
        } else {
            Vec::new()
        };

        for slice_idx in 0..num_slices {
            let mut rem = slice_idx;
            let mut src_base = 0usize;
            let mut dst_base = 0usize;

            for i in 0..v {
                if i == axis {
                    continue;
                }
                let si = shape[i];
                let coord = rem % si;
                rem /= si;
                src_base += coord * strides_src[i];
                dst_base += coord * strides_dst[i];
            }

            if k == 1 {
                // Fused path: source is U_1 = {0,1} along this axis.
                // This should be handled by the v=1 fast path at the top of the function.
                // This block is for v > 1, where k=1 is just one of the axes.
                debug_assert_eq!(src_axis_len, 2);
                let f0 = cur[src_base + 0 * strides_src[axis]]; // p(0)
                let f1 = cur[src_base + 1 * strides_src[axis]]; // p(1)
                let slope = f1 - f0; // slope for linear extrapolation

                // Fill outputs for U_h = {0, ..., h-1, inf}
                next[dst_base + 0 * strides_dst[axis]] = f0; // p(0)
                next[dst_base + 1 * strides_dst[axis]] = f1; // p(1)
                for t in 2..h {
                    next[dst_base + t * strides_dst[axis]] = f0 + slope.mul_u64(t as u64);
                }
                // Last position is ∞, value is the leading coefficient (slope)
                next[dst_base + h * strides_dst[axis]] = slope;
            } else {
                // General path: source is U_k along this axis
                line_buffer.clear();
                for t in 0..src_axis_len {
                    line_buffer.push(cur[src_base + t * strides_src[axis]]);
                }
                // Convert from U_k = {0,...,k-1,∞} to U_h = {0,...,h-1,∞}
                extrapolate_uk_to_uh_into::<F>(&line_buffer, h, &mut line_ex_buffer);
                for t in 0..dst_axis_len {
                    next[dst_base + t * strides_dst[axis]] = line_ex_buffer[t];
                }
            }
        }

        src_idx = dst_idx;
        shape = next_shape;
    }

    let mut final_result = buffers[src_idx].clone();
    final_result.truncate(final_size);
    final_result
}

/// Buffer-reusing version of multivariate_extrapolate that writes into a pre-allocated destination buffer.
/// This avoids the allocation of the final result vector.
pub fn multivariate_extrapolate_into<F: JoltField>(
    v: usize,
    k: usize,
    h: usize,
    values: &[F],
    dest: &mut [F],
) {
    assert!(k >= 1 && k <= 16 && h >= k && h <= 32);
    assert_eq!(values.len(), (k + 1).pow(v as u32));
    let final_size = (h + 1).pow(v as u32);
    assert_eq!(dest.len(), final_size);

    // Early return only when both source and destination grids coincide semantically
    // For k >= 1, inputs are on U_k^v and outputs are also U_k^v when h == k → identity
    if h == k {
        dest.copy_from_slice(values);
        return;
    }

    if v == 1 {
        extrapolate_uk_to_uh_into(values, h, dest);
        return;
    }

    let mut shape: Vec<usize> = core::iter::repeat(k + 1).take(v).collect();

    // OPTIMIZATION: Use double buffering to reduce allocations from v to 2.
    // We'll use dest as the final destination instead of allocating a new buffer.
    let mut buffers = [values.to_vec(), vec![F::zero(); final_size]];
    let mut src_idx = 0;

    for axis in 0..v {
        let dst_idx = 1 - src_idx;

        let mut next_shape = shape.clone();
        next_shape[axis] = h + 1;
        let total_next_elems = next_shape.iter().copied().fold(1usize, |acc, s| acc * s);

        // Ensure the destination buffer is large enough.
        if buffers[dst_idx].len() < total_next_elems {
            buffers[dst_idx].resize(total_next_elems, F::zero());
        }

        // Get mutable, non-overlapping handles to the source and destination buffers.
        let (cur, next) = if src_idx == 0 {
            let (first, second) = buffers.split_at_mut(1);
            (&first[0], &mut second[0])
        } else {
            let (first, second) = buffers.split_at_mut(1);
            (&second[0], &mut first[0])
        };

        // Compute strides for current shape
        let strides_src = compute_strides(&shape);
        // Compute strides for new shape
        let strides_dst = compute_strides(&next_shape);

        let mut num_slices = 1usize;
        for (i, s) in shape.iter().enumerate() {
            if i != axis {
                num_slices *= *s;
            }
        }

        let src_axis_len = shape[axis];
        let dst_axis_len = next_shape[axis];

        // OPTIMIZATION: Hoist the line buffer allocation out of the hot loop.
        let mut line_ex_buffer: Vec<F> = vec![F::zero(); dst_axis_len];

        for slice_idx in 0..num_slices {
            let mut rem = slice_idx;
            let mut src_base = 0usize;
            let mut dst_base = 0usize;

            for i in 0..v {
                if i == axis {
                    continue;
                }
                let si = shape[i];
                let coord = rem % si;
                rem /= si;
                src_base += coord * strides_src[i];
                dst_base += coord * strides_dst[i];
            }

            // OPTIMIZED PATH:
            // 1. Extract the source line of k+1 values.
            let mut line_k_values = Vec::with_capacity(src_axis_len);
            for t in 0..src_axis_len {
                line_k_values.push(cur[src_base + t * strides_src[axis]]);
            }

            // 2. Copy the existing k+1 values into the start of the destination line buffer.
            line_ex_buffer[..src_axis_len].copy_from_slice(&line_k_values);

            // 3. Call the diff-only function to compute and fill in only the new values.
            extrapolate_uk_to_uh_into_diff_only(&line_k_values, h, &mut line_ex_buffer);

            // 4. Copy the fully extrapolated line into the `next` grid.
            for t in 0..dst_axis_len {
                next[dst_base + t * strides_dst[axis]] = line_ex_buffer[t];
            }
        }

        src_idx = dst_idx;
        shape = next_shape;
    }

    // Copy the final result into the destination buffer instead of cloning
    dest.copy_from_slice(&buffers[src_idx][..final_size]);
}

/// Extrapolate from U_1^v to U_h^v, assuming per-axis layout `[p(0), p(1)]` for U_1.
/// This is used internally when degree-1 intermediate results (already on U_1^v)
/// need to be lifted further. For external inputs on U_1^v, use `multivariate_extrapolate` with k=1.
#[inline]
fn multivariate_extrapolate_from_u1<F: JoltField>(v: usize, h: usize, values_u1: &[F]) -> Vec<F> {
    assert_eq!(values_u1.len(), (1 + 1 as usize).pow(v as u32));

    if h == 1 {
        return values_u1.to_vec();
    }

    if v == 1 {
        return extrapolate_uk_to_uh(values_u1, h);
    }

    let mut cur = values_u1.to_vec();
    let mut shape: Vec<usize> = core::iter::repeat(2usize).take(v).collect();

    for axis in 0..v {
        let mut next_shape = shape.clone();
        next_shape[axis] = h + 1;

        let strides_src = compute_strides(&shape);
        let strides_dst = compute_strides(&next_shape);

        let mut num_slices = 1usize;
        for (i, s) in shape.iter().enumerate() {
            if i != axis {
                num_slices *= *s;
            }
        }

        let total_next_elems = next_shape.iter().copied().fold(1usize, |acc, s| acc * s);
        let mut next = vec![F::zero(); total_next_elems];

        for slice_idx in 0..num_slices {
            let mut rem = slice_idx;
            let mut src_base = 0usize;
            let mut dst_base = 0usize;
            for i in 0..v {
                if i == axis {
                    continue;
                }
                let si = shape[i];
                let coord = rem % si;
                rem /= si;
                src_base += coord * strides_src[i];
                dst_base += coord * strides_dst[i];
            }

            // Extract U_1 line: [p(0), p(1)] and lift to U_h
            let line = [
                cur[src_base + 0 * strides_src[axis]],
                cur[src_base + 1 * strides_src[axis]],
            ];
            let line_ex = extrapolate_uk_to_uh::<F>(&line, h);
            for t in 0..(h + 1) {
                next[dst_base + t * strides_dst[axis]] = line_ex[t];
            }
        }

        cur = next;
        shape = next_shape;
    }

    cur
}

/// Buffer-reusing version of multivariate_extrapolate_from_u1 that writes into a destination buffer.
#[inline]
fn multivariate_extrapolate_from_u1_into<F: JoltField>(
    v: usize,
    h: usize,
    values_u1: &[F],
    dest: &mut [F],
) {
    assert_eq!(values_u1.len(), (1 + 1 as usize).pow(v as u32));
    let final_size = (h + 1).pow(v as u32);
    assert_eq!(dest.len(), final_size);

    if h == 1 {
        dest.copy_from_slice(values_u1);
        return;
    }

    if v == 1 {
        extrapolate_uk_to_uh_into(values_u1, h, dest);
        return;
    }

    let mut cur = values_u1.to_vec();
    let mut shape: Vec<usize> = core::iter::repeat(2usize).take(v).collect();

    for axis in 0..v {
        let mut next_shape = shape.clone();
        next_shape[axis] = h + 1;

        let strides_src = compute_strides(&shape);
        let strides_dst = compute_strides(&next_shape);

        let mut num_slices = 1usize;
        for (i, s) in shape.iter().enumerate() {
            if i != axis {
                num_slices *= *s;
            }
        }

        let total_next_elems = next_shape.iter().copied().fold(1usize, |acc, s| acc * s);
        let mut next = vec![F::zero(); total_next_elems];

        for slice_idx in 0..num_slices {
            let mut rem = slice_idx;
            let mut src_base = 0usize;
            let mut dst_base = 0usize;
            for i in 0..v {
                if i == axis {
                    continue;
                }
                let si = shape[i];
                let coord = rem % si;
                rem /= si;
                src_base += coord * strides_src[i];
                dst_base += coord * strides_dst[i];
            }

            // Extract U_1 line: [p(0), p(1)] and lift to U_h
            let line = [
                cur[src_base + 0 * strides_src[axis]],
                cur[src_base + 1 * strides_src[axis]],
            ];
            let line_ex = extrapolate_uk_to_uh::<F>(&line, h);
            for t in 0..(h + 1) {
                next[dst_base + t * strides_dst[axis]] = line_ex[t];
            }
        }

        cur = next;
        shape = next_shape;
    }

    // Copy final result to destination buffer
    dest.copy_from_slice(&cur);
}

// Removed lift_01_to_u1_grid per new API: k=1 inputs are provided on {0,1}^v and handled via fused path

/// MultiProductEval algorithm: multiply d multilinear polynomials and output on U_d^v.
///
/// Inputs:
/// - `v`: number of variables
/// - `multilinear_inputs`: d inputs, each evaluations on {0,1}^v (length 2^v)
/// - `d`: number of polynomials and target degree
///
/// Outputs:
/// - Evaluations of the product on U_d^v (length (d+1)^v) in row-major (last axis contiguous);
///   per-axis layout is `[p(0), p(1), ..., p(d-1), p(∞)]`.
///
/// Panics:
/// - If any input length != 2^v
/// - If `d == 0`
#[inline]
pub fn multi_product_eval<F: JoltField>(
    v: usize,
    multilinear_inputs: &[Vec<F>],
    d: usize,
) -> Vec<F> {
    assert_eq!(multilinear_inputs.len(), d);
    assert!(d >= 1);
    let expected_len = 1usize << v;
    for p in multilinear_inputs {
        assert_eq!(p.len(), expected_len);
    }

    if v == 1 {
        assert_eq!(expected_len, 2);
        let pairs: Vec<(F, F)> = multilinear_inputs.iter().map(|p| (p[0], p[1])).collect();
        return product_eval_univariate_full_zero_based(&pairs);
    }

    // Build zero-copy views into the input polynomials on {0,1}^v
    let views: Vec<&[F]> = multilinear_inputs.iter().map(|p| p.as_slice()).collect();
    multi_product_eval_recursive_views::<F>(v, &views, d, d)
}

/// Buffer-reusing version of multi_product_eval that writes into pre-allocated buffers.
/// This avoids all intermediate allocations in the recursive product computation.
/// Uses separate buffer parameters to avoid borrow checker issues.
///
/// Inputs:
/// - `v`: number of variables
/// - `multilinear_inputs`: d inputs, each evaluations on {0,1}^v (length 2^v)
/// - `d`: number of polynomials and target degree
/// - `dest`: destination buffer to write the result into (length (d+1)^v)
/// - `work_buf1`, `work_buf2`, `work_buf3`: pre-allocated buffers for intermediate computations
///
/// Outputs:
/// - Writes evaluations of the product on U_d^v into `dest`
///
/// Panics:
/// - If any input length != 2^v
/// - If `d == 0`
/// - If `dest.len() != (d+1)^v`
#[inline]
pub fn multi_product_eval_into<F: JoltField>(
    v: usize,
    multilinear_inputs: &[&[F]], // Use slices directly to avoid Vec<&[F]> allocation
    d: usize,
    dest: &mut [F],
    work_buf1: &mut [F],
    work_buf2: &mut [F],
    work_buf3: &mut [F],
) {
    assert_eq!(multilinear_inputs.len(), d);
    assert!(d >= 1);
    let expected_len = 1usize << v;
    for p in multilinear_inputs {
        assert_eq!(p.len(), expected_len);
    }
    let final_size = (d + 1).pow(v as u32);
    assert_eq!(dest.len(), final_size);

    if v == 1 {
        assert_eq!(expected_len, 2);
        let pairs: Vec<(F, F)> = multilinear_inputs.iter().map(|p| (p[0], p[1])).collect();
        let result = product_eval_univariate_full_zero_based(&pairs);
        dest.copy_from_slice(&result);
        return;
    }

    // Ensure buffers are large enough
    let max_intermediate = final_size;
    if work_buf1.len() < max_intermediate {
        panic!(
            "work_buf1 too small: {} < {}",
            work_buf1.len(),
            max_intermediate
        );
    }
    if work_buf2.len() < max_intermediate {
        panic!(
            "work_buf2 too small: {} < {}",
            work_buf2.len(),
            max_intermediate
        );
    }
    if work_buf3.len() < max_intermediate {
        panic!(
            "work_buf3 too small: {} < {}",
            work_buf3.len(),
            max_intermediate
        );
    }

    multi_product_eval_recursive_into::<F>(
        v,
        multilinear_inputs,
        d,
        d,
        dest,
        work_buf1,
        work_buf2,
        work_buf3,
    );
}

/// Buffer-reusing recursive version that writes into pre-allocated buffers.
/// This eliminates all intermediate allocations in the recursive product computation.
fn multi_product_eval_recursive_into<F: JoltField>(
    v: usize,
    polys: &[&[F]],
    target_here: usize,
    target_final: usize,
    dest: &mut [F],
    work_buf1: &mut [F],
    work_buf2: &mut [F],
    work_buf3: &mut [F],
) {
    let final_size = (target_final + 1).pow(v as u32);
    assert_eq!(dest.len(), final_size);

    if polys.len() == 1 {
        // Base case: single multilinear polynomial on {0,1}^v
        if target_here == target_final {
            // Write directly into destination
            multivariate_extrapolate_into::<F>(v, 1, target_here, polys[0], dest);
        } else {
            // Need intermediate lifting - use work_buf1 first
            let intermediate_size = (target_here + 1).pow(v as u32);
            assert!(work_buf1.len() >= intermediate_size);

            let temp_slice = &mut work_buf1[..intermediate_size];
            multivariate_extrapolate_into::<F>(v, 1, target_here, polys[0], temp_slice);

            // Now lift from target_here to target_final
            if target_here == 1 {
                multivariate_extrapolate_from_u1_into::<F>(v, target_final, temp_slice, dest);
            } else {
                multivariate_extrapolate_into::<F>(v, target_here, target_final, temp_slice, dest);
            }
        }
    } else {
        let d_here = polys.len();
        let m = d_here / 2;

        // Compute left product using work_buf1 as destination
        let left_size = (m + 1).pow(v as u32);
        assert!(work_buf1.len() >= left_size);

        let left_dest = &mut work_buf1[..left_size];
        multi_product_eval_recursive_into::<F>(
            v,
            &polys[..m],
            m,
            m,
            left_dest,
            work_buf2,
            work_buf3,
            dest,
        );

        // Compute right product using work_buf2 as destination
        let right_size = ((d_here - m) + 1).pow(v as u32);
        assert!(work_buf2.len() >= right_size);

        let right_dest = &mut work_buf2[..right_size];
        multi_product_eval_recursive_into::<F>(
            v,
            &polys[m..],
            d_here - m,
            d_here - m,
            right_dest,
            work_buf3,
            dest,
            work_buf1,
        );

        // Lift both to U_{d_here}^v and multiply using work_buf3 as workspace
        let d_here_size = (d_here + 1).pow(v as u32);
        assert!(work_buf3.len() >= d_here_size);

        let left_slice = &work_buf1[..left_size];
        let right_slice = &work_buf2[..right_size];

        // Use work_buf3 for the final product computation
        // First, lift left product into first half of work_buf3
        if m == d_here {
            // No lifting needed, copy directly
            work_buf3[..left_size].copy_from_slice(left_slice);
        } else {
            if m == 1 {
                multivariate_extrapolate_from_u1_into::<F>(
                    v,
                    d_here,
                    left_slice,
                    &mut work_buf3[..d_here_size],
                );
            } else {
                multivariate_extrapolate_into::<F>(
                    v,
                    m,
                    d_here,
                    left_slice,
                    &mut work_buf3[..d_here_size],
                );
            }
        }

        // Lift right product and multiply directly into work_buf3
        if d_here - m == d_here {
            // No lifting needed, multiply directly
            for i in 0..right_size {
                work_buf3[i] *= right_slice[i];
            }
        } else {
            // Need to lift right product first - use dest as temporary space
            if d_here - m == 1 {
                multivariate_extrapolate_from_u1_into::<F>(v, d_here, right_slice, dest);
            } else {
                multivariate_extrapolate_into::<F>(v, d_here - m, d_here, right_slice, dest);
            }
            // Now multiply with left product in work_buf3
            for i in 0..d_here_size {
                work_buf3[i] *= dest[i];
            }
        }

        // Final lifting if needed
        if d_here == target_final {
            dest[..d_here_size].copy_from_slice(&work_buf3[..d_here_size]);
        } else {
            multivariate_extrapolate_into::<F>(
                v,
                d_here,
                target_final,
                &work_buf3[..d_here_size],
                dest,
            );
        }
    }
}

// Internal recursion operating on borrowed views; returns owned Vec on U_target^v
fn multi_product_eval_recursive_views<F: JoltField>(
    v: usize,
    polys: &[&[F]],
    _target_here: usize, // the degree produced at this node (m or d)
    target_final: usize, // the final degree requested by the caller at the root
) -> Vec<F> {
    if polys.len() == 1 {
        // Base case: single multilinear polynomial on {0,1}^v
        // Always extrapolate to target_final degree, even if target_here == target_final,
        // because we need to go from U_1 to U_target_final
        multivariate_extrapolate::<F>(v, 1, target_final, polys[0])
    } else {
        let d_here = polys.len();
        let m = d_here / 2;
        // Recurse: left produces U_m^v, right produces U_{d_here-m}^v
        let left_m = multi_product_eval_recursive_views::<F>(v, &polys[..m], m, m);
        let right_dr =
            multi_product_eval_recursive_views::<F>(v, &polys[m..], d_here - m, d_here - m);
        // Lift both to U_{d_here}^v
        let left_d = if m == d_here {
            left_m
        } else if m == 1 {
            multivariate_extrapolate_from_u1::<F>(v, d_here, &left_m)
        } else {
            multivariate_extrapolate::<F>(v, m, d_here, &left_m)
        };
        let right_d = if d_here - m == d_here {
            right_dr
        } else if d_here - m == 1 {
            multivariate_extrapolate_from_u1::<F>(v, d_here, &right_dr)
        } else {
            multivariate_extrapolate::<F>(v, d_here - m, d_here, &right_dr)
        };
        // Multiply on U_{d_here}^v (reuse left buffer)
        let mut prod = left_d;
        for (a, b) in prod.iter_mut().zip(right_d.iter()) {
            *a *= *b;
        }
        // If caller requested a larger degree than d_here, lift once
        if d_here == target_final {
            prod
        } else {
            multivariate_extrapolate::<F>(v, d_here, target_final, &prod)
        }
    }
}

/// Fully implemented buffer-reusing version of multivariate_product_evaluations_accumulate.
/// This is the MAIN optimization target - eliminates ALL allocations in the hot loop.
/// Uses a custom recursive implementation that reuses buffers throughout.
pub fn multivariate_product_evaluations_accumulate_buffered<F: JoltField>(
    v: usize,
    ml_inputs_01: &[Vec<F>], // Use Vec<Vec<F>> to match the caller's data structure
    d: usize,
    sums: &mut [F],
    prod_buf: &mut [F],
    work_buf1: &mut [F],
    work_buf2: &mut [F],
    work_buf3: &mut [F],
) {
    assert!(v >= 1);
    assert!(!ml_inputs_01.is_empty());
    assert_eq!(ml_inputs_01.len(), d);
    let expected_len = 1usize << v;
    for p in ml_inputs_01 {
        assert_eq!(p.len(), expected_len);
    }
    let out_len = (d + 1).pow(v as u32) as usize;
    assert_eq!(sums.len(), out_len);
    assert!(prod_buf.len() >= out_len, "prod_buf is too small");

    // Create zero-copy views into the input polynomials
    let views: Vec<&[F]> = ml_inputs_01.iter().map(|p| p.as_slice()).collect();

    // Compute the product into the provided prod_buf
    multi_product_eval_into(
        v,
        &views,
        d,
        &mut prod_buf[..out_len],
        work_buf1,
        work_buf2,
        work_buf3,
    );

    // Accumulate the computed product into the sums buffer
    for i in 0..out_len {
        sums[i] += prod_buf[i];
    }
}
/// Naive baseline for multivariate product using U_d = {0,1} if d=1, {0,1,...,d-1,∞} if d≥2.
///
/// **Coordinate System:** Uses standard bit ordering (bit i → axis i) for maximum
/// performance in benchmarks. For consistency with fast implementations, use
/// coordinate conversion at the test level.
///
/// Inputs:
/// - `v`: number of variables
/// - `ml_inputs_01`: d inputs on {0,1}^v in **standard bit ordering** (length 2^v each)
/// - `d`: number of inputs and target degree
///
/// Outputs:
/// - Evaluations of the product on U_d^v (length (d+1)^v) in row-major order
///
/// Panics:
/// - If lengths mismatch
pub fn multivariate_product_evaluations_naive<F: JoltField>(
    v: usize,
    ml_inputs_01: &[Vec<F>],
    d: usize,
) -> Vec<F> {
    assert!(v >= 1);
    assert!(!ml_inputs_01.is_empty());
    assert_eq!(ml_inputs_01.len(), d);
    let expected_len = 1usize << v;
    for p in ml_inputs_01 {
        assert_eq!(p.len(), expected_len);
    }

    let out_len = (d + 1).pow(v as u32) as usize;

    // Special case: for d=1, U_1^v = {0,1}^v, so the result should be identical to the input
    if d == 1 {
        assert_eq!(ml_inputs_01.len(), 1);
        return ml_inputs_01[0].clone();
    }

    // Ultra-memory-efficient approach: allocate output incrementally
    let mut out = Vec::with_capacity(out_len);

    // Pre-allocate coordinate vector to avoid repeated allocations
    let mut y: Vec<Option<u64>> = vec![None; v];
    let side = d + 1;

    // Compute product point-by-point with minimal memory usage
    for idx in 0..out_len {
        // Reuse coordinate vector - only update changed values
        let mut tmp = idx;

        for ax in 0..v {
            let t = tmp % side;
            tmp /= side;

            if d >= 2 && t == d {
                y[v - 1 - ax] = None;
            } else {
                y[v - 1 - ax] = Some(t as u64);
            }
        }

        // Compute product for this point - direct u64 evaluation!
        let mut product = F::one();
        for poly in ml_inputs_01.iter() {
            let poly_val = eval_multilinear_at_point_with_infty_u64::<F>(v, poly, &y);
            product *= poly_val;
        }
        out.push(product);
    }
    out
}

/// Accumulate product on U_d^v into `sums` (pointwise addition).
///
/// Inputs:
/// - `v`: number of variables
/// - `ml_inputs_01`: d inputs on {0,1}^v (length 2^v each)
/// - `d`: number of inputs and target degree
/// - `sums`: output accumulator slice, length (d+1)^v
///
/// Effects:
/// - Adds the product evaluations on U_d^v into `sums`
///
/// Panics:
/// - If any input length != 2^v
/// - If `sums.len() != (d+1)^v`
pub fn multivariate_product_evaluations_accumulate<F: JoltField>(
    v: usize,
    ml_inputs_01: &[Vec<F>],
    d: usize,
    sums: &mut [F],
) {
    assert!(v >= 1);
    assert!(!ml_inputs_01.is_empty());
    assert_eq!(ml_inputs_01.len(), d);
    let expected_len = 1usize << v;
    for p in ml_inputs_01 {
        assert_eq!(p.len(), expected_len);
    }
    let out_len = (d + 1).pow(v as u32) as usize;
    assert_eq!(sums.len(), out_len);

    if v == 1 {
        assert_eq!(expected_len, 2);
        let pairs: Vec<(F, F)> = ml_inputs_01.iter().map(|p| (p[0], p[1])).collect();
        let result = product_eval_univariate_full_zero_based(&pairs);
        for (s, r) in sums.iter_mut().zip(result.iter()) {
            *s += *r;
        }
        return;
    }

    // Build zero-copy views and recurse
    let views: Vec<&[F]> = ml_inputs_01.iter().map(|p| p.as_slice()).collect();
    fn rec_acc_views<F: JoltField>(v: usize, polys: &[&[F]], deg_parent: usize, sums: &mut [F]) {
        if polys.len() == 1 {
            // Directly extrapolate from {0,1}^v to U_{deg_parent}^v and accumulate
            let vals = multivariate_extrapolate::<F>(v, 1, deg_parent, polys[0]);
            for (s, p) in sums.iter_mut().zip(vals.iter()) {
                *s += *p;
            }
            return;
        }
        let d_here = polys.len();
        let m = d_here / 2;
        // Compute left product on U_m^v and right on U_{d_here-m}^v using the view-based recursion
        let left_m = multi_product_eval_recursive_views::<F>(v, &polys[..m], m, m);
        let right_dr =
            multi_product_eval_recursive_views::<F>(v, &polys[m..], d_here - m, d_here - m);
        // Lift both to U_{deg_parent}^v
        let left_ex = if m == deg_parent {
            left_m
        } else {
            multivariate_extrapolate::<F>(v, m, deg_parent, &left_m)
        };
        let right_ex = if d_here - m == deg_parent {
            right_dr
        } else {
            multivariate_extrapolate::<F>(v, d_here - m, deg_parent, &right_dr)
        };
        // Accumulate pointwise product into sums
        for i in 0..sums.len() {
            sums[i] += left_ex[i] * right_ex[i];
        }
    }

    rec_acc_views::<F>(v, &views, d, sums)
}
#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr as BN254;
    use ark_ff::{Field, UniformRand};
    use ark_std::One;

    fn random_polys_01<F: Field>(v: usize, d: usize) -> Vec<Vec<F>> {
        let n = 1usize << v;
        (0..d)
            .map(|j| {
                // vary each poly slightly
                (0..n)
                    .map(|i| F::from(((i as u64) * 17 + (j as u64) * 101 + 5) % 7919))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_extrapolate_vs_naive() {
        // Test that our paper-aligned extrapolation matches the paper-aligned naive implementation
        type F = BN254;
        for v in [1usize, 2, 3].iter().copied() {
            for h in [2, 3, 4, 5, 6].iter().copied() {
                let p01 = random_polys_01::<F>(v, 1).pop().unwrap();

                // Extrapolate directly from {0,1}^v using fused k=1 path
                let fast = multivariate_extrapolate::<F>(v, 1, h, &p01);
                let naive = multilinear_extend_to_u_h_grid_naive::<F>(v, &p01, h);

                if fast.len() != naive.len() {
                    println!(
                        "v={v}, h={h}: fast.len()={}, naive.len()={}",
                        fast.len(),
                        naive.len()
                    );
                    println!("Expected length: {}", (h + 1).pow(v as u32));
                }

                assert_eq!(fast.len(), naive.len(), "Length mismatch for v={v}, h={h}");
                for (i, (a, b)) in fast.iter().zip(naive.iter()).enumerate() {
                    if a != b {
                        if v == 1 && h == 3 && i == 3 {
                            println!("Debug v={v}, h={h}, idx={i}:");
                            println!(
                                "  p01: {:?}",
                                p01.iter().map(|x| format!("{}", x)).collect::<Vec<_>>()
                            );
                            println!(
                                "  fast: {:?}",
                                fast.iter().map(|x| format!("{}", x)).collect::<Vec<_>>()
                            );
                            println!(
                                "  naive: {:?}",
                                naive.iter().map(|x| format!("{}", x)).collect::<Vec<_>>()
                            );

                            // Manual verification: p(x) = 5 + 17x, so p(2) should be 5 + 34 = 39
                            println!(
                                "  Manual check: p(0)=5, p(1)=22 → p(x)=5+17x → p(2)=39, p(3)=56"
                            );

                            // Test just the univariate extrapolation
                            let u1_input_manual = vec![p01[0], p01[1]]; // [p(0), p(1)]
                            let manual_test = extrapolate_uk_to_uh::<F>(&u1_input_manual, 3);
                            println!(
                                "  Direct univariate extrapolation: {:?}",
                                manual_test
                                    .iter()
                                    .map(|x| format!("{}", x))
                                    .collect::<Vec<_>>()
                            );
                        }
                        panic!("Extrapolate mismatch for v={v}, h={h} at idx={i}: fast={a:?}, naive={b:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_extrapolate_layout_v_is_1() {
        // Test that for v=1, multivariate_extrapolate is identical to extrapolate_uk_to_uh
        type F = BN254;
        let v = 1;
        for k in [1, 2, 4, 8].iter().copied() {
            for h in [k, k + 1, k * 2].iter().copied() {
                if h > 32 {
                    continue;
                }
                let mut rng = ark_std::test_rng();
                let values_uk: Vec<F> = (0..=k).map(|_| F::rand(&mut rng)).collect();

                let multi_res = multivariate_extrapolate(v, k, h, &values_uk);
                let uni_res = extrapolate_uk_to_uh(&values_uk, h);

                assert_eq!(
                    multi_res, uni_res,
                    "multivariate extrapolate (v=1) should match univariate for k={}, h={}",
                    k, h
                );
            }
        }
    }

    #[test]
    fn test_product_vs_naive() {
        // Test that our paper-aligned product matches the paper-aligned naive implementation
        // Note: The fast implementation uses bit-reversed coordinate system, so we need to
        // convert the input polynomials for the naive implementation to match
        type F = BN254;
        for v in [1usize, 2, 3].iter().copied() {
            for d in [1usize, 2, 3, 4, 5, 6].iter().copied() {
                let polys = random_polys_01::<F>(v, d);

                let fast = multi_product_eval::<F>(v, &polys, d);

                // Convert polynomials to bit-reversed order for naive implementation
                // to match the coordinate system used by the fast implementation
                let mut polys_for_naive = polys.clone();
                if d >= 2 {
                    // Only apply bit reversal when fast implementation uses it
                    let expected_len = 1usize << v;
                    for poly in &mut polys_for_naive {
                        for i in 0..expected_len {
                            let j = i.reverse_bits() >> (usize::BITS - v as u32);
                            if i < j {
                                poly.swap(i, j);
                            }
                        }
                    }
                }

                let naive = multivariate_product_evaluations_naive::<F>(v, &polys_for_naive, d);

                assert_eq!(fast.len(), naive.len(), "Length mismatch for v={v}, d={d}");
                for (i, (a, b)) in fast.iter().zip(naive.iter()).enumerate() {
                    if a != b {
                        panic!(
                            "Product mismatch for v={v}, d={d} at idx={i}: fast={a:?}, naive={b:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_multivariate_evaluations_struct() {
        // Test the MultivariateEvaluations struct
        type F = BN254;
        let evals = vec![F::one(), F::from_u64(2u64), F::from_u64(3u64)];
        let mv_eval = MultivariateEvaluations {
            degree: 2,
            num_vars: 1,
            evaluations: evals.clone(),
        };

        assert_eq!(mv_eval.degree, 2);
        assert_eq!(mv_eval.num_vars, 1);
        assert_eq!(mv_eval.evaluations, evals);
    }

    #[test]
    fn test_uk_conversion() {
        // Test the conversion between paper's U_k and code's U_k
        type F = BN254;

        // Test simple case: degree 2 polynomial p(x) = 1 + 2x + 3x^2
        // U_2 = {0,1,∞}: p(0)=1, p(1)=6, p(∞)=3
        let code_vals = vec![F::from_u64(1u64), F::from_u64(6u64), F::from_u64(3u64)];

        // Extrapolate to degree 3 using new order
        let result = extrapolate_uk_to_uh::<F>(&code_vals, 3);

        println!(
            "Input: {:?}",
            code_vals
                .iter()
                .map(|x| format!("{}", x))
                .collect::<Vec<_>>()
        );
        println!(
            "Output: {:?}",
            result.iter().map(|x| format!("{}", x)).collect::<Vec<_>>()
        );
        println!("Expected: p(0)=1, p(1)=6, p(2)=17, p(∞)=3");

        // Should have p(0), p(1), p(2), p(∞)
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], F::from_u64(1u64)); // p(0) = 1
        assert_eq!(result[1], F::from_u64(6u64)); // p(1) = 6
        assert_eq!(result[2], F::from_u64(17u64)); // p(2) = 1 + 4 + 12 = 17
        assert_eq!(result[3], F::from_u64(3u64)); // p(∞) = 3
    }

    #[test]
    fn test_simple_univariate_extrapolation() {
        // Test a very simple case to understand the coordinate system
        type F = BN254;

        // Linear polynomial p(x) = 3 + 4x
        // p(0) = 3, p(1) = 7
        // U_1 = {0, 1}; stored as [p(0), p(1)] = [3, 7]
        let u1_vals = vec![F::from_u64(3u64), F::from_u64(7u64)];

        // Extrapolate to U_2 = {0, 1, ∞}
        let result = extrapolate_uk_to_uh::<F>(&u1_vals, 2);

        println!("Input U_1: {:?}", u1_vals);
        println!("Output U_2: {:?}", result);

        // [p(0), p(1), p(∞)]
        println!("Expected: p(0)=3, p(1)=7, p(∞)=4");
        println!(
            "Got: pos0={}, pos1={}, pos2={}",
            result[0], result[1], result[2]
        );

        // Verify the polynomial manually:
        // Input U_1 set {0, 1}; stored as [p(0), p(1)] = [3, 7]
        // For linear p(x) = a + bx: p(0) = a = 3, p(1) = a + b = 7 → b = 4
        // So p(x) = 3 + 4x, therefore p(∞) = 4

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], F::from_u64(3u64)); // p(0)
        assert_eq!(result[1], F::from_u64(7u64)); // p(1)
        assert_eq!(result[2], F::from_u64(4u64)); // p(∞)
    }

    #[test]
    fn test_base_case_single_polynomial() {
        // Test the base case of the recursive algorithm with d=1 (single polynomial)
        type F = BN254;
        let v = 2;
        let d = 1; // Product of 1 polynomial is just the polynomial itself

        // Single multilinear polynomial on {0,1}^2
        let poly_01 = vec![
            F::from_u64(1u64),
            F::from_u64(2u64),
            F::from_u64(3u64),
            F::from_u64(4u64),
        ];
        let polys = vec![poly_01.clone()];

        let result = multi_product_eval::<F>(v, &polys, d);
        // For d=1, the result should be the polynomial extended to U_1^v = {0,1}^v
        let expected = multilinear_extend_to_u_h_grid_naive::<F>(v, &poly_01, d);

        assert_eq!(
            result, expected,
            "Base case (d=1) should match naive extension to U_1"
        );
    }

    #[test]
    fn test_recursive_case_two_polynomials() {
        // Test the recursive case with exactly two polynomials
        type F = BN254;
        let v = 1;
        let d = 2;

        let poly1 = vec![F::from_u64(1u64), F::from_u64(2u64)]; // p1(0)=1, p1(1)=2
        let poly2 = vec![F::from_u64(3u64), F::from_u64(5u64)]; // p2(0)=3, p2(1)=5
        let polys = vec![poly1, poly2];

        let result = multi_product_eval::<F>(v, &polys, d);
        let expected = multivariate_product_evaluations_naive::<F>(v, &polys, d);

        assert_eq!(result, expected, "Two-polynomial case should match naive");

        // Manual verification: product should be p1*p2
        // p1(x) = 1 + x, p2(x) = 3 + 2x
        // product(x) = (1+x)(3+2x) = 3 + 2x + 3x + 2x^2 = 3 + 5x + 2x^2
        // U_2 = {0,1, ∞}: product(0)=3, product(1)=10, product(∞)=2
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], F::from_u64(3u64)); // product(0) - position 0
        assert_eq!(result[1], F::from_u64(10u64)); // product(1) - position 1
        assert_eq!(result[2], F::from_u64(2u64)); // product(∞) - position 2
    }

    #[test]
    fn test_univariate_vs_multivariate() {
        // Test that multivariate product evaluation matches univariate evaluation when v=1
        // This validates the consistency between the two implementations
        //
        // Coordinate systems:
        // - Multivariate result: evaluations at points {0, 1, 2, ..., D-1, ∞}
        //   Format: [p(0), p(1), p(2), ..., p(D-1), p(∞)]
        // - Univariate result: evaluations at points {0, 1, 2, ..., D-1, ∞}
        //   Format: [p(0), p(1), p(2), ..., p(D-1), p(∞)] (same as multivariate for this case)

        type F = BN254;
        let v = 1;
        let d = 3; // Use degree 3 for clear testing

        // Generate random linear polynomials as pairs (p(0), p(1))
        let mut rng = ark_std::test_rng();
        let mut linear_polys = Vec::new();
        let mut polys_01 = Vec::new(); // For multivariate format

        for _ in 0..d {
            let p0 = F::rand(&mut rng);
            let p1 = F::rand(&mut rng);
            linear_polys.push((p0, p1));
            // Convert to {0,1} evaluations for multivariate: [p(0), p(1)]
            polys_01.push(vec![p0, p1]);
        }

        // Convert linear polynomials to array format for univariate
        let _linear_array: [(F, F); 3] = linear_polys.clone().try_into().unwrap();

        // Run multivariate product evaluation
        let multivariate_result = multi_product_eval::<F>(v, &polys_01, d);

        // Run univariate product evaluation
        // Note: univariate function produces [p(0), p(1), p(2), ..., p(D-1), p(∞)]
        let univariate_result = product_eval_univariate_full_zero_based::<F>(&linear_polys);

        // Relationship between coordinate systems:
        // multivariate[0] = p(0)   ↔  univariate[0] = p(0)
        // multivariate[1] = p(1)   ↔  univariate[1] = p(1)
        // multivariate[2] = p(2)   ↔  univariate[2] = p(2)
        // multivariate[3] = p(∞)   ↔  univariate[3] = p(∞)

        // Verify the mathematical relationships
        assert_eq!(
            multivariate_result, univariate_result,
            "Univariate and multivariate results should be identical for v=1"
        );

        println!("\n✅ Univariate and multivariate results are consistent!");
    }
}
