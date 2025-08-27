use crate::field::JoltField;
use ark_ff::Field;
use ark_std::vec::Vec;

// Small helpers mirroring high-d-opt
#[inline]
fn dbl<F: JoltField>(x: F) -> F {
    x + x
}

#[inline]
fn mul6<F: JoltField>(x: F) -> F {
    x.mul_u64(6)
}

// -----------------------------------------------------------------------------
// Product of D linear polynomials via recursive evaluation-based approach
// -----------------------------------------------------------------------------
// The `eval_inter*` functions implement the univariate product algorithm from
// sections/4_high_d.tex. The algorithm stays in the evaluation domain.
//
// Strategy (divide-and-conquer):
// 1) Split the D factors in half.
// 2) Recursively form two partial products, each represented by evaluations on U_{⌈D/2⌉}.
// 3) Extrapolate both to U_D.
// 4) Multiply point-wise to obtain the result on U_D.
//
// Implementation details:
// - Hardcoded kernels are provided for D ∈ {2,4,8,16,32}.
// - These kernels work over the 1-based finite grid with ∞ appended: [1,2,...,D, ∞].
//   This is a convenience layout for low-level recurrences. Higher-level APIs convert
//   to and from the paper's U_D = {0,1} if D=1, {0,1,...,D-1,∞} if D≥2 as needed.
// - `_final` variants omit a trailing finite point (used by protocol windows).
// - `_accumulate` variants fuse accumulation into an existing buffer.
// -----------------------------------------------------------------------------

/// Baseline O(D^2) accumulate of the product table on the 1-based grid.
///
/// Inputs:
/// - `pairs`: for each factor j, `(p_j(0), p_j(1))`
/// - `sums`: accumulator with layout [1, 2, ..., D-1, ∞]
///
/// Effects:
/// - Adds g(1..D-1) and g(∞) where g = ∏ p_j
///
/// Precondition:
/// - `sums.len() == if D>1 { D } else { 1 }`
#[inline]
pub fn product_eval_univariate_accumulate_naive<F: JoltField, const D: usize>(
    pairs: &[(F, F); D],
    sums: &mut [F],
) {
    debug_assert_eq!(sums.len(), if D > 1 { D } else { 1 });

    if D == 0 {
        return;
    }
    if D == 1 {
        // g(1)
        sums[0] += pairs[0].1;
        return;
    }

    // Evaluate g(k) for k = 1..D-1
    for k in 1..D {
        let kf = F::from_u64(k as u64);
        let mut prod = F::one();
        for &(a0, a1) in pairs.iter() {
            let slope = a1 - a0;
            let val = a0 + slope * kf;
            prod *= val;
        }
        sums[k - 1] += prod;
    }

    // Evaluate g(∞) = ∏ (a1 - a0)
    let mut prod_inf = F::one();
    for &(a0, a1) in pairs.iter() {
        prod_inf *= a1 - a0;
    }
    sums[D - 1] += prod_inf;
}

// d = 2: seed points [1, 2, ∞] from two linear factors
#[inline]
fn eval_inter2<F: JoltField>((p0, p1): (F, F), (q0, q1): (F, F)) -> (F, F, F) {
    let p_inf = p1 - p0;
    let p2 = p_inf + p1;
    let q_inf = q1 - q0;
    let q2 = q_inf + q1;
    let r1 = p1 * q1;
    let r2 = p2 * q2;
    let r_inf = p_inf * q_inf;
    (r1, r2, r_inf)
}

// d = 2: message subset [1, ∞]
#[inline]
fn eval_inter2_final<F: JoltField>((p0, p1): (F, F), (q0, q1): (F, F)) -> (F, F) {
    let r1 = p1 * q1;
    let r_inf = (p1 - p0) * (q1 - q0);
    (r1, r_inf)
}

// d = 4: extend [1,2,∞] to [1,2,3,4,∞]
#[inline]
fn eval_inter4<F: JoltField>(p: [(F, F); 4]) -> (F, F, F, F, F) {
    #[inline]
    fn helper<F: JoltField>(fx0: F, fx1: F, f_inf: F) -> F {
        dbl(fx1 + f_inf) - fx0
    }
    let (a1, a2, a_inf) = eval_inter2(p[0], p[1]);
    let a3 = helper(a1, a2, a_inf);
    let a4 = helper(a2, a3, a_inf);
    let (b1, b2, b_inf) = eval_inter2(p[2], p[3]);
    let b3 = helper(b1, b2, b_inf);
    let b4 = helper(b2, b3, b_inf);
    (a1 * b1, a2 * b2, a3 * b3, a4 * b4, a_inf * b_inf)
}

// d = 4: final subset [1,2,3,∞]
#[inline]
fn eval_inter4_final<F: JoltField>(p: [(F, F); 4]) -> (F, F, F, F) {
    #[inline]
    fn helper<F: JoltField>(fx0: F, fx1: F, f_inf: F) -> F {
        dbl(fx1 + f_inf) - fx0
    }
    let (a1, a2, a_inf) = eval_inter2(p[0], p[1]);
    let a3 = helper(a1, a2, a_inf);
    let (b1, b2, b_inf) = eval_inter2(p[2], p[3]);
    let b3 = helper(b1, b2, b_inf);
    (a1 * b1, a2 * b2, a3 * b3, a_inf * b_inf)
}

// d = 3: compute [1,2,3,∞] from three linear factors
#[inline]
fn eval_inter3<F: JoltField>(p: [(F, F); 3]) -> [F; 4] {
    #[inline]
    fn helper<F: JoltField>(fx0: F, fx1: F, f_inf: F) -> F {
        // same forward-step as in eval_inter4: f_{t+1} = 2*(f_t + f_inf) - f_{t-1}
        dbl(fx1 + f_inf) - fx0
    }
    // First combine p0 and p1 to get quadratic A on {1,2,inf}, then step to 3
    let (a1, a2, a_inf) = eval_inter2(p[0], p[1]);
    let a3 = helper(a1, a2, a_inf);
    // Third linear factor values at 1,2,3 and inf
    let (c0, c1) = p[2];
    let c_inf = c1 - c0;
    let c2 = c1 + c_inf; // 2*c1 - c0
    let c3 = c2 + c_inf; // 3*c1 - 2*c0
    [a1 * c1, a2 * c2, a3 * c3, a_inf * c_inf]
}

// d = 8: extend [1,2,3,4,∞] to [1..8,∞]
#[inline(always)]
fn eval_inter8<F: JoltField>(p: [(F, F); 8]) -> [F; 9] {
    #[inline]
    fn helper_pair<F: JoltField>(f: &[F; 4], f_inf6: F) -> (F, F) {
        let f3m2 = f[3] - f[2];
        let f4 = dbl(dbl(f_inf6 + f3m2 + f[1]) - f[2]) - f[0];
        let f5 = dbl(dbl(f4 - f3m2 + f_inf6) - f[3]) - f[1];
        (f4, f5)
    }
    #[inline]
    fn batch_helper<F: JoltField>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F, F) {
        let f_inf6 = mul6(f_inf);
        let (f4, f5) = helper_pair(&[f0, f1, f2, f3], f_inf6);
        let (f6, f7) = helper_pair(&[f2, f3, f4, f5], f_inf6);
        (f4, f5, f6, f7)
    }
    let (a1, a2, a3, a4, a_inf) = eval_inter4(unsafe { *(p[0..4].as_ptr() as *const [(F, F); 4]) });
    let (a5, a6, a7, a8) = batch_helper(a1, a2, a3, a4, a_inf);
    let (b1, b2, b3, b4, b_inf) = eval_inter4(unsafe { *(p[4..8].as_ptr() as *const [(F, F); 4]) });
    let (b5, b6, b7, b8) = batch_helper(b1, b2, b3, b4, b_inf);
    [
        a1 * b1,
        a2 * b2,
        a3 * b3,
        a4 * b4,
        a5 * b5,
        a6 * b6,
        a7 * b7,
        a8 * b8,
        a_inf * b_inf,
    ]
}

// d = 8: final subset [1..7,∞]
#[inline]
fn eval_inter8_final<F: JoltField>(p: [(F, F); 8]) -> [F; 8] {
    #[inline]
    fn helper_pair<F: JoltField>(f: &[F; 4], f_inf6: F) -> (F, F) {
        let f3m2 = f[3] - f[2];
        let f4 = dbl(dbl(f_inf6 + f3m2 + f[1]) - f[2]) - f[0];
        let f5 = dbl(dbl(f4 - f3m2 + f_inf6) - f[3]) - f[1];
        (f4, f5)
    }
    #[inline]
    fn batch_helper<F: JoltField>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F) {
        let f_inf6 = mul6(f_inf);
        let (f4, f5) = helper_pair(&[f0, f1, f2, f3], f_inf6);
        let f6 = dbl(dbl(f_inf6 + f5 - f4 + f3) - f4) - f2;
        (f4, f5, f6)
    }
    let (a1, a2, a3, a4, a_inf) = eval_inter4(unsafe { *(p[0..4].as_ptr() as *const [(F, F); 4]) });
    let (a5, a6, a7) = batch_helper(a1, a2, a3, a4, a_inf);
    let (b1, b2, b3, b4, b_inf) = eval_inter4(unsafe { *(p[4..8].as_ptr() as *const [(F, F); 4]) });
    let (b5, b6, b7) = batch_helper(b1, b2, b3, b4, b_inf);
    [
        a1 * b1,
        a2 * b2,
        a3 * b3,
        a4 * b4,
        a5 * b5,
        a6 * b6,
        a7 * b7,
        a_inf * b_inf,
    ]
}

// Fused accumulate variants for layout [1, 2, ..., D-1, ∞]
#[inline(always)]
fn eval_inter2_final_accumulate<F: JoltField>(p0: (F, F), p1: (F, F), sums: &mut [F]) {
    let r1 = p0.1 * p1.1; // g(1)
    let r_inf = (p0.1 - p0.0) * (p1.1 - p1.0); // ∞
    sums[0] += r1;
    sums[1] += r_inf;
}

#[inline(always)]
fn eval_inter4_final_accumulate<F: JoltField>(p: &[(F, F); 4], sums: &mut [F]) {
    #[inline]
    fn helper<F: JoltField>(fx0: F, fx1: F, f_inf: F) -> F {
        dbl(fx1 + f_inf) - fx0
    }
    let (a1, a2, a_inf) = eval_inter2(p[0], p[1]);
    let a3 = helper(a1, a2, a_inf);
    let (b1, b2, b_inf) = eval_inter2(p[2], p[3]);
    let b3 = helper(b1, b2, b_inf);
    sums[0] += a1 * b1; // 1
    sums[1] += a2 * b2; // 2
    sums[2] += a3 * b3; // 3
    sums[3] += a_inf * b_inf; // ∞
}

#[inline(always)]
fn eval_inter8_final_accumulate<F: JoltField>(p: &[(F, F); 8], sums: &mut [F]) {
    // A direct port of the logic from eval_inter8_final, but accumulating.
    #[inline]
    fn helper_pair<F: JoltField>(f: &[F; 4], f_inf6: F) -> (F, F) {
        let f3m2 = f[3] - f[2];
        let f4 = dbl(dbl(f_inf6 + f3m2 + f[1]) - f[2]) - f[0];
        let f5 = dbl(dbl(f4 - f3m2 + f_inf6) - f[3]) - f[1];
        (f4, f5)
    }
    #[inline]
    fn batch_helper<F: JoltField>(f0: F, f1: F, f2: F, f3: F, f_inf: F) -> (F, F, F) {
        let f_inf6 = mul6(f_inf);
        let (f4, f5) = helper_pair(&[f0, f1, f2, f3], f_inf6);
        let f6 = dbl(dbl(f_inf6 + f5 - f4 + f3) - f4) - f2;
        (f4, f5, f6)
    }
    let (a1, a2, a3, a4, a_inf) = eval_inter4(unsafe { *(p[0..4].as_ptr() as *const [(F, F); 4]) });
    let (a5, a6, a7) = batch_helper(a1, a2, a3, a4, a_inf);
    let (b1, b2, b3, b4, b_inf) = eval_inter4(unsafe { *(p[4..8].as_ptr() as *const [(F, F); 4]) });
    let (b5, b6, b7) = batch_helper(b1, b2, b3, b4, b_inf);

    sums[0] += a1 * b1;
    sums[1] += a2 * b2;
    sums[2] += a3 * b3;
    sums[3] += a4 * b4;
    sums[4] += a5 * b5;
    sums[5] += a6 * b6;
    sums[6] += a7 * b7;
    sums[7] += a_inf * b_inf;
}

// d = 16: extend [1..8,∞] to [1..16,∞]
#[inline(always)]
fn eval_inter16<F: JoltField>(p: [(F, F); 16]) -> [F; 17] {
    #[inline]
    fn helper_idx<F: JoltField>(f: &[F; 17], b: usize, f_inf40320: F) -> F {
        // window is f[b-8..b]
        let a1 = f[b - 7] + f[b - 1]; // f[1]+f[7]
        let a2 = f[b - 5] + f[b - 3]; // f[3]+f[5]
        let n1 = f[b - 6] + f[b - 2]; // f[2]+f[6]
        let n2 = f[b - 4]; // f[4]
        let n3 = f[b - 8]; // f[0]
        F::linear_combination_i64(
            &[(a1, 8), (a2, 56), (f_inf40320, 1)],
            &[(n1, 28), (n2, 70), (n3, 1)],
        )
    }
    #[inline]
    // #[unroll_for_loops(8)]
    fn batch_helper<F: JoltField>(vals: &[F; 9]) -> [F; 17] {
        let mut f = [F::zero(); 17]; // f[1, ..., 16, inf]
        for i in 0..8 {
            f[i] = vals[i];
        }
        f[16] = vals[8];
        let f_inf40320 = vals[8].mul_u64(40320); // 8!
        for i in 8..16 {
            f[i] = helper_idx(&f, i, f_inf40320);
        }
        f
    }
    let mut res = batch_helper(&eval_inter8(unsafe {
        *(p[0..8].as_ptr() as *const [(F, F); 8])
    }));
    let aux = batch_helper(&eval_inter8(unsafe {
        *(p[8..16].as_ptr() as *const [(F, F); 8])
    }));
    for i in 0..17 {
        res[i] *= aux[i];
    }
    res
}

// d = 16: final subset [1..15,∞]
#[inline]
// #[unroll_for_loops(16)]
fn eval_inter16_final<F: JoltField>(p: [(F, F); 16]) -> [F; 16] {
    #[inline]
    fn helper_idx<F: JoltField>(f: &[F; 16], b: usize, f_inf40320: F) -> F {
        // window is f[b-8..b]
        let a1 = f[b - 7] + f[b - 1]; // f[1]+f[7]
        let a2 = f[b - 5] + f[b - 3]; // f[3]+f[5]
        let n1 = f[b - 6] + f[b - 2]; // f[2]+f[6]
        let n2 = f[b - 4]; // f[4]
        let n3 = f[b - 8]; // f[0]
        F::linear_combination_i64(
            &[(a1, 8), (a2, 56), (f_inf40320, 1)],
            &[(n1, 28), (n2, 70), (n3, 1)],
        )
    }
    #[inline]
    // #[unroll_for_loops(8)]
    fn batch_helper<F: JoltField>(vals: &[F; 9]) -> [F; 16] {
        let mut f = [F::zero(); 16]; // f[1, ..., 15, inf]
        for i in 0..8 {
            f[i] = vals[i];
        }
        f[15] = vals[8];
        let f_inf40320 = vals[8].mul_u64(40320);
        for i in 8..15 {
            f[i] = helper_idx(&f, i, f_inf40320);
        }
        f
    }
    let mut res = batch_helper(&eval_inter8(unsafe {
        *(p[0..8].as_ptr() as *const [(F, F); 8])
    }));
    let aux = batch_helper(&eval_inter8(unsafe {
        *(p[8..16].as_ptr() as *const [(F, F); 8])
    }));
    for i in 0..16 {
        res[i] *= aux[i];
    }
    res
}

#[inline(always)]
// #[unroll_for_loops(16)]
fn eval_inter16_final_accumulate<F: JoltField>(p: &[(F, F); 16], sums: &mut [F]) {
    #[inline]
    fn helper_idx<F: JoltField>(f: &[F; 16], b: usize, f_inf40320: F) -> F {
        // window is f[b-8..b]
        let a1 = f[b - 7] + f[b - 1]; // f[1]+f[7]
        let a2 = f[b - 5] + f[b - 3]; // f[3]+f[5]
        let n1 = f[b - 6] + f[b - 2]; // f[2]+f[6]
        let n2 = f[b - 4]; // f[4]
        let n3 = f[b - 8]; // f[0]
        F::linear_combination_i64(
            &[(a1, 8), (a2, 56), (f_inf40320, 1)],
            &[(n1, 28), (n2, 70), (n3, 1)],
        )
    }
    #[inline]
    fn batch_values<F: JoltField>(vals: &[F; 9]) -> [F; 16] {
        let mut f = [F::zero(); 16]; // f[1..15, inf]
        for i in 0..8 {
            f[i] = vals[i];
        }
        f[15] = vals[8];
        let f_inf40320 = vals[8].mul_u64(40320);
        for i in 8..15 {
            f[i] = helper_idx(&f, i, f_inf40320);
        }
        f
    }
    let a = eval_inter8(unsafe { *(p[0..8].as_ptr() as *const [(F, F); 8]) });
    let b = eval_inter8(unsafe { *(p[8..16].as_ptr() as *const [(F, F); 8]) });
    let mut av = batch_values(&a);
    let bv = batch_values(&b);
    // Include all entries [1..15, inf]
    for i in 0..16 {
        av[i] *= bv[i];
        sums[i] += av[i];
    }
}

// d = 32: extend [1..16,∞] to [1..32,∞]
#[inline]
#[allow(dead_code)]
fn eval_inter32<F: JoltField>(p: [(F, F); 32]) -> [F; 33] {
    #[inline]
    fn helper<F: JoltField>(f: &[F; 16], f_infbig: F) -> F {
        F::linear_combination_i64(
            &[
                (f[1] + f[15], 16),
                (f[3] + f[13], 560),
                (f[5] + f[11], 4368),
                (f[7] + f[9], 11440),
                (f_infbig, 1),
            ],
            &[
                (f[2] + f[14], 120),
                (f[4] + f[12], 1820),
                (f[6] + f[10], 8008),
                (f[8], 12870),
                (f[0], 1),
            ],
        )
    }
    #[inline]
    fn batch_helper<F: JoltField>(vals: &[F; 17]) -> [F; 33] {
        let mut f = [F::zero(); 33]; // f[1, ..., 32, inf]
        for i in 0..16 {
            f[i] = vals[i];
        }
        f[32] = vals[16];
        let f_infbig = vals[16].mul_u64(20922789888000u64); // 16!
        for i in 16..32 {
            f[i] = helper(
                unsafe { &*(f[(i - 16)..i].as_ptr() as *const [F; 16]) },
                f_infbig,
            );
        }
        f
    }
    let a = batch_helper(&eval_inter16(unsafe {
        *(p[0..16].as_ptr() as *const [(F, F); 16])
    }));
    let b = batch_helper(&eval_inter16(unsafe {
        *(p[16..32].as_ptr() as *const [(F, F); 16])
    }));
    let mut res = [F::zero(); 33];
    for i in 0..33 {
        res[i] = a[i] * b[i];
    }
    res
}

// d = 32: final subset [1..31,∞]
#[inline]
fn eval_inter32_final<F: JoltField>(p: [(F, F); 32]) -> [F; 32] {
    #[inline]
    fn helper<F: JoltField>(f: &[F; 16], f_infbig: F) -> F {
        F::linear_combination_i64(
            &[
                (f[1] + f[15], 16),
                (f[3] + f[13], 560),
                (f[5] + f[11], 4368),
                (f[7] + f[9], 11440),
                (f_infbig, 1),
            ],
            &[
                (f[2] + f[14], 120),
                (f[4] + f[12], 1820),
                (f[6] + f[10], 8008),
                (f[8], 12870),
                (f[0], 1),
            ],
        )
    }
    #[inline]
    fn batch_helper<F: JoltField>(vals: &[F; 17]) -> [F; 32] {
        let mut f = [F::zero(); 32]; // f[1, ..., 31, inf]
        for i in 0..16 {
            f[i] = vals[i];
        }
        f[31] = vals[16];
        let f_infbig = vals[16].mul_u64(20922789888000u64);
        for i in 16..31 {
            f[i] = helper(
                unsafe { &*(f[(i - 16)..i].as_ptr() as *const [F; 16]) },
                f_infbig,
            );
        }
        f
    }
    let mut a = batch_helper(&eval_inter16(unsafe {
        *(p[0..16].as_ptr() as *const [(F, F); 16])
    }));
    let b = batch_helper(&eval_inter16(unsafe {
        *(p[16..32].as_ptr() as *const [(F, F); 16])
    }));
    for i in 0..32 {
        a[i] *= b[i];
    }
    a
}

#[inline(always)]
fn eval_inter32_final_accumulate<F: JoltField>(p: &[(F, F); 32], sums: &mut [F]) {
    #[inline]
    fn helper<F: JoltField>(f: &[F; 16], f_infbig: F) -> F {
        F::linear_combination_i64(
            &[
                (f[1] + f[15], 16),
                (f[3] + f[13], 560),
                (f[5] + f[11], 4368),
                (f[7] + f[9], 11440),
                (f_infbig, 1),
            ],
            &[
                (f[2] + f[14], 120),
                (f[4] + f[12], 1820),
                (f[6] + f[10], 8008),
                (f[8], 12870),
                (f[0], 1),
            ],
        )
    }
    #[inline]
    // // #[unroll_for_loops(16)]
    fn batch_values<F: JoltField>(vals: &[F; 17]) -> [F; 32] {
        let mut f = [F::zero(); 32]; // f[1..31, inf]
        for i in 0..16 {
            f[i] = vals[i];
        }
        f[31] = vals[16];
        let f_infbig = vals[16].mul_u64(20922789888000u64);
        for i in 16..31 {
            f[i] = helper(
                unsafe { &*(f[(i - 16)..i].as_ptr() as *const [F; 16]) },
                f_infbig,
            );
        }
        f
    }
    let a = eval_inter16(unsafe { *(p[0..16].as_ptr() as *const [(F, F); 16]) });
    let b = eval_inter16(unsafe { *(p[16..32].as_ptr() as *const [(F, F); 16]) });
    let mut av = batch_values(&a);
    let bv = batch_values(&b);
    // Include all entries [1..31, inf]
    for i in 0..32 {
        av[i] *= bv[i];
        sums[i] += av[i];
    }
}

// -----------------------------------------------------------------------------
// Univariate extrapolation kernels over the 1-based grid [1..k, ∞]
// -----------------------------------------------------------------------------
// The `extend*` functions implement the optimized univariate extrapolation from
// sections/D_high_d_appendix.tex for the convenience grid [1,2,...,k, ∞].
// They extend to [1..2k, ∞] using small-integer linear combinations.
//
// These are internal helpers. The paper-aligned API is `extrapolate_uk_to_uh`, which
// consumes and produces evaluations on U_k = {0,1} if k=1, {0,1,...,k-1,∞} if k≥2
// It converts to the [1..k,∞] layout by shifting x↦x+1, calls the kernels, then shifts back.
// -----------------------------------------------------------------------------

#[allow(dead_code)]
#[inline]
fn extend1_to_2<F: JoltField>(vals: &[F; 2]) -> [F; 3] {
    let f1 = vals[0];
    let f_inf = vals[1];
    [f1, f1 + f_inf, f_inf]
}

/// Preconditions:
/// - f is a univariate polynomial of degree exactly 2, so f(∞) equals the
///   leading coefficient.
/// - vals holds a consecutive block with unit spacing: [f(a), f(a+1), f(∞)].
///
/// Effect (shift-invariant):
/// - Produces [f(a), f(a+1), f(a+2), f(a+3), f(∞)]. The formulas depend only
///   on unit spacing, not on the absolute start index a.
#[allow(dead_code)]
#[inline]
fn extend2_to_4<F: JoltField>(vals: &[F; 3]) -> [F; 5] {
    let f1 = vals[0];
    let f2 = vals[1];
    let f_inf = vals[2];
    let f3 = -f1 + dbl(f2 + f_inf); // -1*f1 + 2*f2 + 2*f_inf
    let f4 = -(dbl(f1)) + (f2 + f2 + f2) + f_inf.mul_u64(6); // -2*f1 + 3*f2 + 6*f_inf
    [f1, f2, f3, f4, f_inf]
}

/// Extend from 2 points to 4, returning only the new evaluations [f3, f4].
/// This is more efficient when you only need the additional points.
#[inline]
fn extend2_to_4_diff_only<F: JoltField>(vals: &[F; 3]) -> [F; 2] {
    let f1 = vals[0];
    let f2 = vals[1];
    let f_inf = vals[2];
    let f3 = -f1 + dbl(f2 + f_inf);
    let f4 = -(dbl(f1)) + (f2 + f2 + f2) + f_inf.mul_u64(6);
    [f3, f4]
}

/// Preconditions:
/// - f is a univariate polynomial of degree exactly 4, so f(∞) equals the
///   leading coefficient.
/// - vals holds a consecutive block with unit spacing: [f(a), f(a+1), f(a+2), f(a+3), f(∞)].
///
/// Effect (shift-invariant):
/// - Produces [f(a), f(a+1), ..., f(a+7), f(∞)]. The formulas depend only on
///   unit spacing, not on the absolute start index a.
#[allow(dead_code)]
#[inline]
fn extend4_to_8<F: JoltField>(vals: &[F; 5]) -> [F; 9] {
    #[inline]
    fn helper_pair<F: JoltField>(f: &[F; 4], f_inf6: F) -> (F, F) {
        let f3m2 = f[3] - f[2];
        let f4 = dbl(dbl(f_inf6 + f3m2 + f[1]) - f[2]) - f[0];
        let f5 = dbl(dbl(f4 - f3m2 + f_inf6) - f[3]) - f[1];
        (f4, f5)
    }
    let f1 = vals[0];
    let f2 = vals[1];
    let f3 = vals[2];
    let f4 = vals[3];
    let f_inf = vals[4];
    let f_inf6 = mul6(f_inf);
    let (f5, f6) = helper_pair(&[f1, f2, f3, f4], f_inf6);
    let (f7, f8) = helper_pair(&[f3, f4, f5, f6], f_inf6);
    [f1, f2, f3, f4, f5, f6, f7, f8, f_inf]
}

/// Extend from 4 points to 8, returning only the new evaluations [f5, f6, f7, f8].
/// This is more efficient when you only need the additional points.
#[inline]
fn extend4_to_8_diff_only<F: JoltField>(vals: &[F; 5]) -> [F; 4] {
    #[inline]
    fn helper_pair<F: JoltField>(f: &[F; 4], f_inf6: F) -> (F, F) {
        let f3m2 = f[3] - f[2];
        let f4 = dbl(dbl(f_inf6 + f3m2 + f[1]) - f[2]) - f[0];
        let f5 = dbl(dbl(f4 - f3m2 + f_inf6) - f[3]) - f[1];
        (f4, f5)
    }
    let f1 = vals[0];
    let f2 = vals[1];
    let f3 = vals[2];
    let f4 = vals[3];
    let f_inf = vals[4];
    let f_inf6 = mul6(f_inf);
    let (f5, f6) = helper_pair(&[f1, f2, f3, f4], f_inf6);
    let (f7, f8) = helper_pair(&[f3, f4, f5, f6], f_inf6);
    [f5, f6, f7, f8]
}

/// Preconditions:
/// - f is a univariate polynomial of degree exactly 8, so f(∞) equals the
///   leading coefficient.
/// - vals holds a consecutive block with unit spacing: [f(a), ..., f(a+7), f(∞)].
///
/// Effect (shift-invariant):
/// - Produces [f(a), ..., f(a+15), f(∞)]. The coefficients use fixed small
///   integers and are independent of the absolute start index a.
#[allow(dead_code)]
#[inline]
fn extend8_to_16<F: JoltField>(vals: &[F; 9]) -> [F; 17] {
    #[inline]
    fn helper<F: JoltField>(f: &[F; 8], f_inf40320: F) -> F {
        F::linear_combination_i64(
            &[(f[1] + f[7], 8), (f[3] + f[5], 56), (f_inf40320, 1)],
            &[(f[2] + f[6], 28), (f[4], 70), (f[0], 1)],
        )
    }
    let mut f = [F::zero(); 17];
    for i in 0..8 {
        f[i] = vals[i];
    }
    f[16] = vals[8];
    let f_inf40320 = vals[8].mul_u64(40320);
    for i in 8..16 {
        f[i] = helper(
            unsafe { &*(f[(i - 8)..i].as_ptr() as *const [F; 8]) },
            f_inf40320,
        );
    }
    f
}

/// Extend from 8 points to 16, returning only the new evaluations [f9..f16].
/// This is more efficient when you only need the additional points.
#[inline]
fn extend8_to_16_diff_only<F: JoltField>(vals: &[F; 9]) -> [F; 8] {
    #[inline]
    fn helper<F: JoltField>(f: &[F; 8], f_inf40320: F) -> F {
        F::linear_combination_i64(
            &[(f[1] + f[7], 8), (f[3] + f[5], 56), (f_inf40320, 1)],
            &[(f[2] + f[6], 28), (f[4], 70), (f[0], 1)],
        )
    }
    let mut f = [F::zero(); 16];
    for i in 0..8 {
        f[i] = vals[i];
    }
    let f_inf40320 = vals[8].mul_u64(40320);
    for i in 8..16 {
        f[i] = helper(
            unsafe { &*(f[(i - 8)..i].as_ptr() as *const [F; 8]) },
            f_inf40320,
        );
    }
    [f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15]]
}

/// Preconditions:
/// - f is a univariate polynomial of degree exactly 16, so f(∞) equals the
///   leading coefficient.
/// - vals holds a consecutive block with unit spacing: [f(a), ..., f(a+15), f(∞)].
///
/// Effect (shift-invariant):
/// - Produces [f(a), ..., f(a+31), f(∞)]. The integer-coefficient formulas
///   assume unit spacing and are independent of the absolute start index a.
#[allow(dead_code)]
#[inline]
fn extend16_to_32<F: JoltField>(vals: &[F; 17]) -> [F; 33] {
    #[inline]
    fn helper<F: JoltField>(f: &[F; 16], f_infbig: F) -> F {
        F::linear_combination_i64(
            &[
                (f[1] + f[15], 16),
                (f[3] + f[13], 560),
                (f[5] + f[11], 4368),
                (f[7] + f[9], 11440),
                (f_infbig, 1),
            ],
            &[
                (f[2] + f[14], 120),
                (f[4] + f[12], 1820),
                (f[6] + f[10], 8008),
                (f[8], 12870),
                (f[0], 1),
            ],
        )
    }
    let mut f = [F::zero(); 33];
    for i in 0..16 {
        f[i] = vals[i];
    }
    f[32] = vals[16];
    let f_infbig = vals[16].mul_u64(20922789888000u64);
    for i in 16..32 {
        f[i] = helper(
            unsafe { &*(f[(i - 16)..i].as_ptr() as *const [F; 16]) },
            f_infbig,
        );
    }
    f
}

/// Extend from 16 points to 32, returning only the new evaluations [f17..f32].
/// This is more efficient when you only need the additional points.
#[inline]
fn extend16_to_32_diff_only<F: JoltField>(vals: &[F; 17]) -> [F; 16] {
    #[inline]
    fn helper<F: JoltField>(f: &[F; 16], f_infbig: F) -> F {
        F::linear_combination_i64(
            &[
                (f[1] + f[15], 16),
                (f[3] + f[13], 560),
                (f[5] + f[11], 4368),
                (f[7] + f[9], 11440),
                (f_infbig, 1),
            ],
            &[
                (f[2] + f[14], 120),
                (f[4] + f[12], 1820),
                (f[6] + f[10], 8008),
                (f[8], 12870),
                (f[0], 1),
            ],
        )
    }
    let mut f = [F::zero(); 32];
    for i in 0..16 {
        f[i] = vals[i];
    }
    let f_infbig = vals[16].mul_u64(20922789888000u64);
    for i in 16..32 {
        f[i] = helper(
            unsafe { &*(f[(i - 16)..i].as_ptr() as *const [F; 16]) },
            f_infbig,
        );
    }
    [
        f[16], f[17], f[18], f[19], f[20], f[21], f[22], f[23], f[24], f[25], f[26], f[27], f[28],
        f[29], f[30], f[31],
    ]
}

/// Generic function to extend from k points to 2k, returning only the new evaluations.
/// This is more efficient when you only need the additional points.
///
/// Inputs:
/// - `vals`: [f(0), f(1), ..., f(k-1), f(∞)] where k is a power of 2
/// - `k`: current number of points (must be 1, 2, 4, 8, or 16)
///
/// Outputs:
/// - [f(k), f(k+1), ..., f(2k-1)] - only the new evaluations
///
/// Panics:
/// - If k is not a supported power of 2
#[inline]
pub fn extend_k_to_2k_diff_only<F: JoltField>(vals: &[F], k: usize) -> Vec<F> {
    match k {
        1 => {
            let f1 = vals[0];
            let f_inf = vals[1];
            vec![f1 + f_inf]
        }
        2 => {
            let arr: &[F; 3] = unsafe { &*(vals.as_ptr() as *const [F; 3]) };
            extend2_to_4_diff_only(arr).to_vec()
        }
        4 => {
            let arr: &[F; 5] = unsafe { &*(vals.as_ptr() as *const [F; 5]) };
            extend4_to_8_diff_only(arr).to_vec()
        }
        8 => {
            let arr: &[F; 9] = unsafe { &*(vals.as_ptr() as *const [F; 9]) };
            extend8_to_16_diff_only(arr).to_vec()
        }
        16 => {
            let arr: &[F; 17] = unsafe { &*(vals.as_ptr() as *const [F; 17]) };
            extend16_to_32_diff_only(arr).to_vec()
        }
        _ => panic!("unsupported k: {}, must be a power of 2 ≤ 16", k),
    }
}

/// Compute the full table of the product of D linear polynomials on U_D with 0-based indexing.
///
/// Inputs:
/// - `pairs[j] = (p_j(0), p_j(1))` for D factors
///
/// Outputs:
/// - [g(0), g(1), ..., g(D-1), g(∞)] where g = ∏ p_j
///
/// Notes:
/// - Output ordering is [0..D-1, ∞] (0-based grid)
/// - This differs from product_eval_univariate_accumulate which uses [1..D-1, ∞] (1-based grid)
pub fn product_eval_univariate_full_zero_based<F: JoltField>(pairs: &[(F, F)]) -> Vec<F> {
    let d = pairs.len();
    assert!(d >= 1, "need at least one linear factor");
    // Fast paths: convert from 1-based grid [1..D, ∞] to 0-based grid [0..D-1, ∞]
    if d == 1 {
        // For d=1: U_1 = {0,1}, so we need [g(0), g(1)]
        let a0 = pairs[0].0;
        let a1 = pairs[0].1;
        return vec![a0, a1];
    }
    if d == 2 {
        let arr: [(F, F); 2] = unsafe { *(pairs.as_ptr() as *const [(F, F); 2]) };
        let (r1, r_inf) = eval_inter2_final(arr[0], arr[1]); // [g(1), g(∞)]
        let g0 = arr[0].0 * arr[1].0; // g(0)
        return vec![g0, r1, r_inf]; // [g(0), g(1), g(∞)]
    }
    if d == 3 {
        let arr: [(F, F); 3] = unsafe { *(pairs.as_ptr() as *const [(F, F); 3]) };
        let r = eval_inter3(arr); // [g(1), g(2), g(3), g(∞)]
        let mut g0 = F::one();
        for (a0, _) in arr.iter().copied() {
            g0 *= a0;
        }
        return vec![g0, r[0], r[1], r[3]]; // [g(0), g(1), g(2), g(∞)]
    }
    if d == 4 {
        let arr: [(F, F); 4] = unsafe { *(pairs.as_ptr() as *const [(F, F); 4]) };
        let (r1, r2, r3, r_inf) = eval_inter4_final(arr); // [g(1), g(2), g(3), g(∞)]
        let mut g0 = F::one();
        for (a0, _) in arr.iter().copied() {
            g0 *= a0;
        }
        return vec![g0, r1, r2, r3, r_inf]; // [g(0), g(1), g(2), g(3), g(∞)]
    }
    if d == 8 {
        let arr: [(F, F); 8] = unsafe { *(pairs.as_ptr() as *const [(F, F); 8]) };
        let r = eval_inter8_final(arr); // [g(1)..g(7), g(∞)]
        let mut g0 = F::one();
        for (a0, _) in arr.iter().copied() {
            g0 *= a0;
        }
        let mut out = vec![F::zero(); 9];
        out[0] = g0; // g(0)
        for i in 0..7 {
            out[1 + i] = r[i];
        } // g(1)..g(7)
        out[8] = r[7]; // g(∞)
        return out;
    }
    if d == 16 {
        let arr: [(F, F); 16] = unsafe { *(pairs.as_ptr() as *const [(F, F); 16]) };
        let r = eval_inter16_final(arr); // [g(1)..g(15), g(∞)]
        let mut g0 = F::one();
        for (a0, _) in arr.iter().copied() {
            g0 *= a0;
        }
        let mut out = vec![F::zero(); 17];
        out[0] = g0; // g(0)
        for i in 0..15 {
            out[1 + i] = r[i];
        } // g(1)..g(15)
        out[16] = r[15]; // g(∞)
        return out;
    }
    if d == 32 {
        let arr: [(F, F); 32] = unsafe { *(pairs.as_ptr() as *const [(F, F); 32]) };
        let r = eval_inter32_final(arr); // [g(1)..g(31), g(∞)]
        let mut g0 = F::one();
        for (a0, _) in arr.iter().copied() {
            g0 *= a0;
        }
        let mut out = vec![F::zero(); 33];
        out[0] = g0; // g(0)
        for i in 0..31 {
            out[1 + i] = r[i];
        } // g(1)..g(31)
        out[32] = r[31]; // g(∞)
        return out;
    }

    // Fallback (non power-of-two): divide & conquer with template-based extrapolation where possible
    fn rec<F: JoltField>(pairs: &[(F, F)]) -> Vec<F> {
        let k = pairs.len();
        if k == 1 {
            // For U_1 = {0,1}, we need [p(0), p(1)]
            let a0 = pairs[0].0;
            let a1 = pairs[0].1;
            return vec![a0, a1];
        }
        let mid = k / 2;
        let left = rec::<F>(&pairs[..mid]);
        let right = rec::<F>(&pairs[mid..]);
        // Extrapolate to U_k
        let left_ext = extrapolate_uk_to_uh::<F>(&left, k);
        let right_ext = extrapolate_uk_to_uh::<F>(&right, k);
        let mut out = vec![F::zero(); k + 1];
        for i in 0..=k {
            out[i] = left_ext[i] * right_ext[i];
        }
        out
    }

    rec::<F>(pairs)
}

/// Fused variant: accumulate the product table on the 1-based grid.
///
/// Inputs:
/// - `pairs[j] = (p_j(0), p_j(1))`
/// - `sums`: accumulator with layout [1, 2, ..., D-1, ∞]
///
/// Effects:
/// - Adds g(1..D-1) and g(∞) into `sums`
///
/// Precondition:
/// - `sums.len() == if D>1 { D } else { 1 }`
#[inline(always)]
pub fn product_eval_univariate_accumulate<F: JoltField, const D: usize>(
    pairs: &[(F, F); D],
    sums: &mut [F],
) {
    debug_assert_eq!(sums.len(), if D > 1 { D } else { 1 });

    match D {
        1 => {
            let (_a0, a1) = pairs[0];
            sums[0] += a1; // g(1)
            return;
        }
        // Fast paths for small D using evaluation-based Toom-style formulas
        2 => {
            // [1, ∞]
            eval_inter2_final_accumulate(pairs[0], pairs[1], sums);
            return;
        }
        3 => {
            let (a0, a1) = pairs[0];
            let (b0, b1) = pairs[1];
            // quadratic A on {0,1,∞}
            let a0q = a0 * b0;
            let a1q = a1 * b1;
            let a_infq = (a1 - a0) * (b1 - b0);
            // extend A to -1
            let a0_2 = a0q + a0q;
            let a_inf_2 = a_infq + a_infq;
            let a_m1 = a0_2 - a1q + a_inf_2;
            // third linear
            let (c0, c1) = pairs[2];
            let c_inf = c1 - c0;
            let c_m1 = c0 - c_inf; // 2*c0 - c1
                                   // point-wise products
            let _r0 = a0q * c0;
            let r1 = a1q * c1; // g(1)
            let r_m1 = a_m1 * c_m1;
            let r_inf = a_infq * c_inf;
            // derive r2 from {0,1,-1,∞}
            let r2 = -_r0.mul_u64(3) + r1.mul_u64(3) + r_m1 + r_inf.mul_u64(6);
            // [1, 2, ∞]
            sums[0] += r1;
            sums[1] += r2;
            sums[2] += r_inf;
            return;
        }
        4 => {
            // [1, 2, 3, ∞]
            let arr: &[(F, F); 4] = unsafe { &*(pairs as *const _ as *const [(F, F); 4]) };
            eval_inter4_final_accumulate(arr, sums);
            return;
        }
        // Fast paths for larger power-of-two D using improved univariate evaluation
        8 => {
            // [1..7, ∞]
            let arr: &[(F, F); 8] = unsafe { &*(pairs as *const _ as *const [(F, F); 8]) };
            eval_inter8_final_accumulate(arr, sums);
            return;
        }
        16 => {
            let arr: &[(F, F); 16] = unsafe { &*(pairs as *const _ as *const [(F, F); 16]) };
            eval_inter16_final_accumulate(arr, sums);
            return;
        }
        32 => {
            let arr: &[(F, F); 32] = unsafe { &*(pairs as *const _ as *const [(F, F); 32]) };
            eval_inter32_final_accumulate(arr, sums);
            return;
        }
        _ => {}
    }

    // Recursive helper returning values on U_k for the subproblem size k.
    fn rec<F: JoltField>(pairs: &[(F, F)]) -> Vec<F> {
        let k = pairs.len();
        if k == 1 {
            let a0 = pairs[0].0;
            let slope = pairs[0].1 - a0;
            let mut out = vec![F::zero(); 1 + 1];
            out[0] = a0; // x=0
            out[1] = slope; // ∞
            return out;
        }
        let mid = k / 2;
        let left = rec::<F>(&pairs[..mid]);
        let right = rec::<F>(&pairs[mid..]);
        let left_ext = extrapolate_uk_to_uh::<F>(&left, k);
        let right_ext = extrapolate_uk_to_uh::<F>(&right, k);
        let mut out = vec![F::zero(); k + 1];
        for i in 0..=k {
            out[i] = left_ext[i] * right_ext[i];
        }
        out
    }

    // Compute subproducts and extrapolate to U_D, then accumulate without materializing `out`.
    let mid = D / 2;
    let left = rec::<F>(&pairs[..mid]);
    let right = rec::<F>(&pairs[mid..]);
    let left_ext = extrapolate_uk_to_uh::<F>(&left, D);
    let right_ext = extrapolate_uk_to_uh::<F>(&right, D);

    // Accumulate message entries: [1, 2, ..., D-1, ∞]
    for k in 1..D {
        sums[k - 1] += left_ext[k] * right_ext[k];
    }
    sums[D - 1] += left_ext[D] * right_ext[D];
}

// -----------------------------------------------------------------------------
// General univariate extrapolation utilities (moved from multivariate.rs)
// -----------------------------------------------------------------------------

/// Univariate extrapolation (generic nodes) using barycentric interpolation.
///
/// Inputs:
/// - `x_nodes`, `x_values`: k distinct nodes and their values
/// - `y_nodes`: target nodes (may include values from `x_nodes`)
///
/// Outputs:
/// - Values at `y_nodes`, consistent with the degree-≤k interpolant through (`x_nodes`,`x_values`)
///
/// Panics:
/// - If `x_nodes.len() != x_values.len()`
pub fn univariate_extrapolate_node_generic<F: Field>(
    x_nodes: &[F],
    x_values: &[F],
    y_nodes: &[F],
) -> Vec<F> {
    assert_eq!(
        x_nodes.len(),
        x_values.len(),
        "nodes/values length mismatch"
    );
    let k = x_nodes.len();
    assert!(k >= 2, "need at least 2 points");

    // Precompute barycentric weights w_i = 1 / prod_{j!=i} (x_i - x_j)
    let mut weights: Vec<F> = vec![F::zero(); k];
    for i in 0..k {
        let mut denom = F::one();
        let xi = x_nodes[i];
        for j in 0..k {
            if i == j {
                continue;
            }
            denom *= xi - x_nodes[j];
        }
        weights[i] = denom.inverse().expect("distinct x_nodes required");
    }

    // Evaluate using first-form barycentric interpolation
    let mut out = Vec::with_capacity(y_nodes.len());
    for &y in y_nodes.iter() {
        // Check if y equals an existing node to avoid division by zero
        let mut at_node: Option<usize> = None;
        for i in 0..k {
            if y == x_nodes[i] {
                at_node = Some(i);
                break;
            }
        }
        if let Some(i) = at_node {
            out.push(x_values[i]);
            continue;
        }

        let mut num = F::zero();
        let mut den = F::zero();
        for i in 0..k {
            let li = weights[i] / (y - x_nodes[i]);
            num += li * x_values[i];
            den += li;
        }
        out.push(num * den.inverse().expect("nonzero denominator"));
    }
    out
}

// -----------------------------------------------------------------------------
// Univariate extrapolation: U_k = {0,1} if k=1, {0,1,...,k-1,∞} if k≥2 → U_h = {0,1,...,h-1,∞}
// -----------------------------------------------------------------------------

/// Univariate extrapolation: U_k = {0,1} if k=1, {0,1,...,k-1,∞} if k≥2 → U_h.
///
/// Inputs:
/// - `values_uk`: [p(0), p(1)] if k=1, [p(0), p(1), ..., p(k-1), p(∞)] if k≥2
/// - `h`: target degree (k ≤ h ≤ 32)
///
/// Outputs:
/// - [p(0), p(1), ..., p(h-1), p(∞)] with the same ordering convention
///
/// Panics:
/// - If `k < 1`, `h < k`, or `h > 32`
#[inline(always)]
pub fn extrapolate_uk_to_uh<F: JoltField>(values_uk: &[F], h: usize) -> Vec<F> {
    let k = values_uk.len() - 1;
    assert!(k >= 1 && h >= k && h <= 32);
    let mut out = vec![F::zero(); h + 1];
    extrapolate_uk_to_uh_into(values_uk, h, &mut out);
    out
}

/// Univariate extrapolation: U_k = {0,1} if k=1, {0,1,...,k-1,∞} if k≥2 → U_h, writing into an output buffer.
/// Diff-only version: assumes out[0..k] already contains the original values,
/// and only writes the new values out[k..=h].
///
/// Inputs:
/// - `values_uk`: [p(0), p(1)] if k=1, [p(0), p(1), ..., p(k-1), p(∞)] if k≥2
/// - `h`: target degree (k ≤ h ≤ 32)
/// - `out`: mutable slice of length h+1, where out[0..k] already contains values_uk[0..k]
///
/// Output:
/// - Writes only out[k..h] (new values p(k) through p(h-1)) and out[h] = p(∞)
///
/// Panics:
/// - If `k < 1`, `h < k`, `h > 32`, or `out.len() != h+1`
#[inline(always)]
pub fn extrapolate_uk_to_uh_into_diff_only<F: JoltField>(values_uk: &[F], h: usize, out: &mut [F]) {
    let k = values_uk.len() - 1;
    assert!(k >= 1 && h >= k && h <= 32);
    assert_eq!(out.len(), h + 1);

    let p_inf = values_uk[k];
    out[h] = p_inf; // Always write p(∞) at the end

    if k == h {
        return; // Nothing new to compute
    }

    // Fast path: k == 1 using linear extrapolation p(t) = p(0) + (p(1) - p(0)) * t
    if k == 1 {
        let p0 = values_uk[0];
        let p1 = values_uk[1];
        let slope = p1 - p0;
        out[0] = p0;
        out[1] = p1;
        for t in 2..h {
            out[t] = out[t - 1] + slope;
        }
        out[h] = slope;
        return;
    }

    // Fast path: k == 2 using second-order recurrence with known leading coeff (p(∞))
    if k == 2 {
        let p_inf = values_uk[2];
        // The recurrence p_{t+1} = 2*(p_t + p_inf) - p_{t-1} is a special case of the
        // general finite difference formula for k=2.
        let mut prev = out[0];
        let mut cur = out[1];
        for t in 2..h {
            let next = dbl(cur + p_inf) - prev;
            out[t] = next;
            prev = cur;
            cur = next;
        }
        return;
    }

    // Fast path: k == 4 by seeding f1..f4 and then using doubling kernels
    if k == 4 {
        let p_inf = values_uk[4];

        // Compute p(4) efficiently using the direct formula from finite differences,
        // avoiding a slow Vandermonde solve.
        // p(4) = 4*p(3) - 6*p(2) + 4*p(1) - p(0) + 4!*p(∞)
        let p0 = out[0];
        let p1 = out[1];
        let p2 = out[2];
        let p3 = out[3];
        let p4 = p3.mul_u64(4) - p2.mul_u64(6) + p1.mul_u64(4) - p0 + p_inf.mul_u64(24);
        out[4] = p4;

        if h == 5 {
            out[h] = p_inf; // Set p(∞) before returning
            return;
        }

        // Extend to 8 using diff_only
        let inp48 = [out[1], out[2], out[3], out[4], p_inf]; // [f1, f2, f3, f4, f∞]
        let out48 = extend4_to_8_diff_only(&inp48); // [f5, f6, f7, f8]
        let limit1 = core::cmp::min(h, 8);
        for t in 5..=limit1 {
            out[t] = out48[t - 5];
        }
        if h <= 8 {
            out[h] = p_inf; // Set p(∞) before returning
            return;
        }

        // Extend to 16
        let inp816 = [
            out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], p_inf,
        ];
        let out816 = extend8_to_16_diff_only(&inp816); // [f9..f16]
        let limit2 = core::cmp::min(h, 16);
        for t in 9..=limit2 {
            out[t] = out816[t - 9];
        }
        if h <= 16 {
            out[h] = p_inf; // Set p(∞) before returning
            return;
        }

        // Extend to 32
        let inp1632 = [
            out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10],
            out[11], out[12], out[13], out[14], out[15], out[16], p_inf,
        ];
        let out1632 = extend16_to_32_diff_only(&inp1632); // [f17..f32]
        for t in 17..=h {
            out[t] = out1632[t - 17];
        }
        out[h] = p_inf; // Set p(∞) at the end
        return;
    }

    // General k>1: robust fallback using finite difference recurrence.
    extrapolate_with_recurrence_into_diff_only(values_uk, h, out)
}

/// Univariate extrapolation: U_k = {0,1} if k=1, {0,1,...,k-1,∞} if k≥2 → U_h, writing into an output buffer.
///
/// Inputs:
/// - `values_uk`: [p(0), p(1)] if k=1, [p(0), p(1), ..., p(k-1), p(∞)] if k≥2
/// - `h`: target degree (k ≤ h ≤ 32)
/// - `out`: mutable slice of length h+1 to store [p(0), p(1), ..., p(h-1), p(∞)]
///
/// Panics:
/// - If `k < 1`, `h < k`, `h > 32`, or `out.len() != h+1`
#[inline(always)]
pub fn extrapolate_uk_to_uh_into<F: JoltField>(values_uk: &[F], h: usize, out: &mut [F]) {
    let k = values_uk.len() - 1;
    assert!(k >= 1 && h >= k && h <= 32);
    assert_eq!(out.len(), h + 1);

    if k == h {
        out.copy_from_slice(values_uk);
        return;
    }

    // Fast path: k == 1 using linear extrapolation p(t) = p(0) + (p(1) - p(0)) * t
    if k == 1 {
        let p0 = values_uk[0];
        let p1 = values_uk[1];
        let slope = p1 - p0;
        out[0] = p0;
        out[1] = p1;
        for t in 2..h {
            out[t] = out[t - 1] + slope;
        }
        out[h] = slope;
        return;
    }

    // Fast path: k == 2 using second-order recurrence with known leading coeff (p(∞))
    if k == 2 {
        let p0 = values_uk[0];
        let p1 = values_uk[1];
        let p_inf = values_uk[2];
        if h == 2 {
            out.copy_from_slice(values_uk);
            return;
        }
        out[0] = p0; // 0
        out[1] = p1; // 1
                     // The recurrence p_{t+1} = 2*(p_t + p_inf) - p_{t-1} is a special case of the
                     // general finite difference formula for k=2.
        let mut prev = p0;
        let mut cur = p1;
        for t in 2..h {
            let next = dbl(cur + p_inf) - prev;
            out[t] = next;
            prev = cur;
            cur = next;
        }
        out[h] = p_inf; // ∞
        return;
    }

    // Fast path: k == 4 by seeding f1..f4 and then using doubling kernels
    if k == 4 {
        let p0 = values_uk[0];
        let p1 = values_uk[1];
        let p2 = values_uk[2];
        let p3 = values_uk[3];
        let p_inf = values_uk[4];
        if h == 4 {
            out.copy_from_slice(values_uk);
            return;
        }

        // Compute p(4) efficiently using the direct formula from finite differences,
        // avoiding a slow Vandermonde solve.
        // p(4) = 4*p(3) - 6*p(2) + 4*p(1) - p(0) + 4!*p(∞)
        let p4 = p3.mul_u64(4) - p2.mul_u64(6) + p1.mul_u64(4) - p0 + p_inf.mul_u64(24);
        let f1 = p1;
        let f2 = p2;
        let f3 = p3;
        let f4 = p4;

        out[0] = p0; // 0
        out[1] = f1; // 1
        out[2] = f2; // 2
        out[3] = f3; // 3
        out[4] = f4; // 4
        if h == 5 {
            out[h] = p_inf;
            return;
        }

        // Extend to 8
        let out48 = extend4_to_8::<F>(&[f1, f2, f3, f4, p_inf]); // [f1..f8,f_inf]
        let limit1 = core::cmp::min(h, 8);
        for t in 5..=limit1 {
            out[t] = out48[t - 1];
        }
        if h <= 8 {
            out[h] = p_inf;
            return;
        }

        // Extend to 16
        let mut inp816 = [F::zero(); 9];
        inp816[..8].copy_from_slice(&out48[..8]);
        inp816[8] = p_inf;
        let out816 = extend8_to_16::<F>(&inp816);
        let limit2 = core::cmp::min(h, 16);
        for t in 9..=limit2 {
            out[t] = out816[t - 1];
        }
        if h <= 16 {
            out[h] = p_inf;
            return;
        }

        // Extend to 32
        let mut inp1632 = [F::zero(); 17];
        inp1632[..16].copy_from_slice(&out816[..16]);
        inp1632[16] = p_inf;
        let out1632 = extend16_to_32::<F>(&inp1632);
        for t in 17..=h {
            out[t] = out1632[t - 1];
        }
        out[h] = p_inf;
        return;
    }

    // General k>1: robust fallback using finite difference recurrence.
    extrapolate_with_recurrence_into(values_uk, h, out)
}

/// Extrapolate from U_k to U_h using recurrence relation derived from finite differences.
/// Diff-only version: assumes out[0..k] already contains the original values.
/// Computes and writes only p(k)...p(h-1), and sets out[h] = p(infinity).
#[inline(always)]
fn extrapolate_with_recurrence_into_diff_only<F: JoltField>(
    values_uk: &[F],
    h: usize,
    out: &mut [F],
) {
    let k = values_uk.len() - 1;
    if h <= k {
        out[h] = values_uk[k]; // Only write p(∞)
        return;
    }
    assert!(k <= 16, "recurrence extrapolation only supported for k<=16");

    let p_inf = values_uk[k];
    let constant_term = p_inf.mul_u64(FACTORIALS[k]);

    let binom_coeffs: &[u64] = &BINOMIAL_COEFFS[k - 1];

    for t in k..h {
        // p(t) = sum_{i=1 to k} (-1)^{i-1} * C(k,i) * p(t-i) + k! * p(inf)
        // t-i ranges from t-1 down to t-k.
        // In `out`, p(x) is at index x. So p(t-i) is at out[t-i].
        let mut sum = F::zero();
        for i in 1..=k {
            let term = out[t - i].mul_u64(binom_coeffs[i - 1]);
            if (i - 1) % 2 == 0 {
                // (-1)^{i-1} is positive
                sum += term;
            } else {
                // (-1)^{i-1} is negative
                sum -= term;
            }
        }
        sum += constant_term;
        out[t] = sum;
    }

    out[h] = p_inf; // Add p(infinity) at the end
}

/// Extrapolate from U_k to U_h using recurrence relation derived from finite differences.
/// Assumes p(0)...p(k-1) are known, and p(infinity) is the leading coefficient.
/// Computes p(k)...p(h-1).
#[inline(always)]
fn extrapolate_with_recurrence_into<F: JoltField>(values_uk: &[F], h: usize, out: &mut [F]) {
    let k = values_uk.len() - 1;
    if h <= k {
        out[..h + 1].copy_from_slice(&values_uk[..h + 1]);
        return;
    }
    assert!(k <= 16, "recurrence extrapolation only supported for k<=16");

    out[..k].copy_from_slice(&values_uk[..k]);

    let p_inf = values_uk[k];
    let constant_term = p_inf.mul_u64(FACTORIALS[k]);

    let binom_coeffs: &[u64] = &BINOMIAL_COEFFS[k - 1];

    for t in k..h {
        // p(t) = sum_{i=1 to k} (-1)^{i-1} * C(k,i) * p(t-i) + k! * p(inf)
        // t-i ranges from t-1 down to t-k.
        // In `out`, p(x) is at index x. So p(t-i) is at out[t-i].
        let mut sum = F::zero();
        for i in 1..=k {
            let term = out[t - i].mul_u64(binom_coeffs[i - 1]);
            if (i - 1) % 2 == 0 {
                // (-1)^{i-1} is positive
                sum += term;
            } else {
                // (-1)^{i-1} is negative
                sum -= term;
            }
        }
        sum += constant_term;
        out[t] = sum;
    }

    out[h] = p_inf; // Add p(infinity) at the end
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr as BN254;
    use ark_ff::AdditiveGroup;
    use ark_std::{One, Zero};

    #[test]
    fn test_extrapolate_uk_to_uh_fast_paths() {
        // Test k=1 fast path: U_k = {0,1} with [p(0), p(1)]
        let u1_vals = vec![BN254::from(3u64), BN254::from(7u64)]; // [p(0), p(1)]
        let result = extrapolate_uk_to_uh::<BN254>(&u1_vals, 3);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], BN254::from(3u64)); // p(0)
        assert_eq!(result[1], BN254::from(7u64)); // p(1)
        assert_eq!(result[2], BN254::from(11u64)); // p(2) = 3 + 2*(7-3) = 3 + 2*4 = 11
        assert_eq!(result[3], BN254::from(15u64)); // p(3) = 3 + 3*(7-3) = 3 + 3*4 = 15

        // Test k=2 fast path
        let u2_vals = vec![BN254::from(1u64), BN254::from(6u64), BN254::from(3u64)]; // [p(0), p(1), p(∞)]
        let result = extrapolate_uk_to_uh::<BN254>(&u2_vals, 4);
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], BN254::from(1u64)); // p(0)
        assert_eq!(result[1], BN254::from(6u64)); // p(1)
        assert_eq!(result[2], BN254::from(17u64)); // p(2) = 2*6 - 1 + 2*3 = 12 - 1 + 6 = 17
        assert_eq!(result[3], BN254::from(34u64)); // p(3) = 2*(17 + 3) - 6 = 2*20 - 6 = 34
        assert_eq!(result[4], BN254::from(3u64)); // p(∞)

        // Test k=4 fast path
        let u4_vals = vec![
            BN254::from(1u64),
            BN254::from(2u64),
            BN254::from(4u64),
            BN254::from(8u64),
            BN254::from(16u64),
        ]; // [p(0), p(1), p(2), p(3), p(∞)]
        let result = extrapolate_uk_to_uh::<BN254>(&u4_vals, 6);
        assert_eq!(result.len(), 7);
        assert_eq!(result[0], BN254::from(1u64)); // p(0)
        assert_eq!(result[1], BN254::from(2u64)); // p(1)
        assert_eq!(result[2], BN254::from(4u64)); // p(2)
        assert_eq!(result[3], BN254::from(8u64)); // p(3)
                                                  // p(4) and p(5) are computed via the doubling kernels
        assert_eq!(result[6], BN254::from(16u64)); // p(∞)
    }

    #[test]
    fn test_extrapolate_uk_to_uh_diff_only() {
        // Test k=1 diff_only fast path: U_k = {0,1} with [p(0), p(1)]
        let u1_vals = vec![BN254::from(3u64), BN254::from(7u64)]; // [p(0), p(1)]
        let mut out = vec![BN254::zero(); 4];
        out[0] = u1_vals[0];
        out[1] = u1_vals[1]; // Pre-fill with existing values
        extrapolate_uk_to_uh_into_diff_only(&u1_vals, 3, &mut out);
        assert_eq!(out[0], BN254::from(3u64)); // p(0) - unchanged
        assert_eq!(out[1], BN254::from(7u64)); // p(1) - unchanged
        assert_eq!(out[2], BN254::from(11u64)); // p(2) = 3 + 2*(7-3) = 3 + 8 = 11
        assert_eq!(out[3], BN254::from(15u64)); // p(3) = 3 + 3*(7-3) = 3 + 12 = 15

        // Test k=2 diff_only fast path
        let u2_vals = vec![BN254::from(1u64), BN254::from(6u64), BN254::from(3u64)]; // [p(0), p(1), p(∞)]
        let mut out = vec![BN254::zero(); 5];
        out[0] = u2_vals[0];
        out[1] = u2_vals[1]; // Pre-fill with existing values
        extrapolate_uk_to_uh_into_diff_only(&u2_vals, 4, &mut out);
        assert_eq!(out[0], BN254::from(1u64)); // p(0) - unchanged
        assert_eq!(out[1], BN254::from(6u64)); // p(1) - unchanged
        assert_eq!(out[2], BN254::from(17u64)); // p(2) = 2*6 - 1 + 2*3 = 12 - 1 + 6 = 17
        assert_eq!(out[3], BN254::from(34u64)); // p(3) = 2*(17 + 3) - 6 = 2*20 - 6 = 34
        assert_eq!(out[4], BN254::from(3u64)); // p(∞)

        // Test k=4 diff_only fast path
        let u4_vals = vec![
            BN254::from(1u64),
            BN254::from(2u64),
            BN254::from(4u64),
            BN254::from(8u64),
            BN254::from(16u64),
        ]; // [p(0), p(1), p(2), p(3), p(∞)]
        let mut out = vec![BN254::zero(); 7];
        out[0] = u4_vals[0];
        out[1] = u4_vals[1];
        out[2] = u4_vals[2];
        out[3] = u4_vals[3]; // Pre-fill with existing values
        extrapolate_uk_to_uh_into_diff_only(&u4_vals, 6, &mut out);
        assert_eq!(out[0], BN254::from(1u64)); // p(0) - unchanged
        assert_eq!(out[1], BN254::from(2u64)); // p(1) - unchanged
        assert_eq!(out[2], BN254::from(4u64)); // p(2) - unchanged
        assert_eq!(out[3], BN254::from(8u64)); // p(3) - unchanged
                                               // p(4), p(5) are computed via the doubling kernels
        assert_eq!(out[6], BN254::from(16u64)); // p(∞)

        // Verify it matches the regular version
        let regular_result = extrapolate_uk_to_uh::<BN254>(&u4_vals, 6);
        assert_eq!(out, regular_result);
    }

    #[test]
    fn test_extend_diff_only_functions() {
        // Test extend2_to_4_diff_only
        let vals = [BN254::from(1u64), BN254::from(2u64), BN254::from(3u64)]; // [f(1), f(2), f(∞)]
        let new_vals = extend2_to_4_diff_only(&vals);
        assert_eq!(new_vals.len(), 2);
        // f(3) = -f(1) + 2*(f(2) + f(∞)) = -1 + 2*(2 + 3) = -1 + 10 = 9
        // f(4) = -2*f(1) + 3*f(2) + 6*f(∞) = -2 + 6 + 18 = 22
        assert_eq!(new_vals[0], BN254::from(9u64)); // f(3)
        assert_eq!(new_vals[1], BN254::from(22u64)); // f(4)

        // Test extend4_to_8_diff_only
        let vals = [
            BN254::from(1u64),
            BN254::from(2u64),
            BN254::from(4u64),
            BN254::from(8u64),
            BN254::from(16u64),
        ]; // [f(1), f(2), f(3), f(4), f(∞)]
        let new_vals = extend4_to_8_diff_only(&vals);
        assert_eq!(new_vals.len(), 4);
        // Verify we get 4 new values
        assert!(new_vals.iter().all(|&x| x != BN254::zero()));

        // Test generic function
        let vals = [BN254::from(1u64), BN254::from(2u64), BN254::from(3u64)]; // [f(1), f(2), f(∞)]
        let new_vals = extend_k_to_2k_diff_only(&vals, 2);
        assert_eq!(new_vals.len(), 2);
        assert_eq!(new_vals[0], BN254::from(9u64)); // f(3)
        assert_eq!(new_vals[1], BN254::from(22u64)); // f(4)
    }
}

const BINOMIAL_COEFFS: [[u64; 16]; 16] = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // k=1
    [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // k=2
    [3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // k=3
    [4, 6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // k=4
    [5, 10, 10, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // k=5
    [6, 15, 20, 15, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], // k=6
    [7, 21, 35, 35, 21, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], // k=7
    [8, 28, 56, 70, 56, 28, 8, 1, 0, 0, 0, 0, 0, 0, 0, 0], // k=8
    [9, 36, 84, 126, 126, 84, 36, 9, 1, 0, 0, 0, 0, 0, 0, 0], // k=9
    [10, 45, 120, 210, 252, 210, 120, 45, 10, 1, 0, 0, 0, 0, 0, 0], // k=10
    [
        11, 55, 165, 330, 462, 462, 330, 165, 55, 11, 1, 0, 0, 0, 0, 0,
    ], // k=11
    [
        12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1, 0, 0, 0, 0,
    ], // k=12
    [
        13, 78, 286, 715, 1287, 1716, 1716, 1287, 715, 286, 78, 13, 1, 0, 0, 0,
    ], // k=13
    [
        14, 91, 364, 1001, 2002, 3003, 3432, 3003, 2002, 1001, 364, 91, 14, 1, 0, 0,
    ], // k=14
    [
        15, 105, 455, 1365, 3003, 5005, 6435, 6435, 5005, 3003, 1365, 455, 105, 15, 1, 0,
    ], // k=15
    [
        16, 120, 560, 1820, 4368, 8008, 11440, 12870, 11440, 8008, 4368, 1820, 560, 120, 16, 1,
    ], // k=16
];
const FACTORIALS: [u64; 17] = [
    1,
    1,
    2,
    6,
    24,
    120,
    720,
    5040,
    40320,
    362880,
    3628800,
    39916800,
    479001600,
    6227020800,
    87178291200,
    1307674368000,
    20922789888000,
];
