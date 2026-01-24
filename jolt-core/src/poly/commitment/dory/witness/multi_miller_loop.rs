//! Multi-Miller loop witness generation for Dory recursion (BN254).
//!
//! This module generates a **packed 11-var trace** over:
//! - step vars `s ∈ {0,1}^7` (up to 128 steps)
//! - element vars `x ∈ {0,1}^4` (16 evaluations for the Fq12 MLE representation)
//!
//! Layout: `idx = x * 128 + s` (step index in the low bits), matching the packed GT exp layout.
//!
//! Important: This witness currently implements a **per-pair Miller loop trace** (one trace per
//! (G1,G2) pair), suitable for decomposing a multi-pairing into multiple Miller loops plus GT muls.

use ark_bn254::{Config as Bn254Config, Fq, Fq12, Fq2, Fq6, G1Affine, G2Affine};
use ark_ec::{bn::BnConfig, AffineRepr};
use ark_ff::{Field, One, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use jolt_optimizations::{fq12_to_multilinear_evals, get_g_mle};

const NUM_STEP_VARS: usize = 7;
const NUM_ELEMENT_VARS: usize = 4;
const STEP_SIZE: usize = 1 << NUM_STEP_VARS; // 128
const ELEM_SIZE: usize = 1 << NUM_ELEMENT_VARS; // 16
const PACKED_SIZE: usize = STEP_SIZE * ELEM_SIZE; // 2048

fn fq12_sparse_034(c0: Fq2, c3: Fq2, c4: Fq2) -> Fq12 {
    // Fq12 = c0 + c1 * w where c0,c1 ∈ Fq6
    // Fq6 = a0 + a1 * v + a2 * v^2 where ai ∈ Fq2
    //
    // `mul_by_034(c0,c3,c4)` corresponds to multiplying by:
    //   (c0, 0, 0) + (c3, c4, 0) * w
    Fq12::new(
        Fq6::new(c0, Fq2::zero(), Fq2::zero()),
        Fq6::new(c3, c4, Fq2::zero()),
    )
}

fn pack_step_and_elem(step_mles: &[Vec<Fq>], num_steps: usize) -> Vec<Fq> {
    debug_assert!(num_steps <= STEP_SIZE);
    let mut packed = vec![Fq::zero(); PACKED_SIZE];
    for s in 0..num_steps.min(STEP_SIZE) {
        let row = &step_mles[s];
        debug_assert_eq!(row.len(), ELEM_SIZE, "expected 16 evals per Fq12 MLE row");
        for x in 0..ELEM_SIZE {
            packed[x * STEP_SIZE + s] = row[x];
        }
    }
    packed
}

fn pack_step_only(step_vals: &[Fq], num_steps: usize) -> Vec<Fq> {
    debug_assert!(num_steps <= STEP_SIZE);
    let mut packed = vec![Fq::zero(); PACKED_SIZE];
    for s in 0..num_steps.min(STEP_SIZE) {
        let v = step_vals[s];
        for x in 0..ELEM_SIZE {
            packed[x * STEP_SIZE + s] = v;
        }
    }
    packed
}

fn pack_step_only_fq2_c0(step_vals: &[Fq2], num_steps: usize) -> Vec<Fq> {
    let step_vals_c0: Vec<Fq> = step_vals.iter().map(|v| v.c0).collect();
    pack_step_only(&step_vals_c0, num_steps)
}

fn pack_step_only_fq2_c1(step_vals: &[Fq2], num_steps: usize) -> Vec<Fq> {
    let step_vals_c1: Vec<Fq> = step_vals.iter().map(|v| v.c1).collect();
    pack_step_only(&step_vals_c1, num_steps)
}

/// Per-pair Miller loop trace, packed into 11-var MLEs.
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct MultiMillerLoopSteps {
    pub g1_points: Vec<G1Affine>,
    pub g2_points: Vec<G2Affine>,

    /// Product of per-pair Miller loop outputs (in Fq12, before final exponentiation).
    pub result: Fq12,

    /// Packed polynomials (one per pair).
    pub f_packed_mles: Vec<Vec<Fq>>,
    pub f_next_packed_mles: Vec<Vec<Fq>>,
    pub quotient_packed_mles: Vec<Vec<Fq>>,

    pub t_x_c0_packed_mles: Vec<Vec<Fq>>,
    pub t_x_c1_packed_mles: Vec<Vec<Fq>>,
    pub t_y_c0_packed_mles: Vec<Vec<Fq>>,
    pub t_y_c1_packed_mles: Vec<Vec<Fq>>,
    pub t_x_c0_next_packed_mles: Vec<Vec<Fq>>,
    pub t_x_c1_next_packed_mles: Vec<Vec<Fq>>,
    pub t_y_c0_next_packed_mles: Vec<Vec<Fq>>,
    pub t_y_c1_next_packed_mles: Vec<Vec<Fq>>,

    pub lambda_c0_packed_mles: Vec<Vec<Fq>>,
    pub lambda_c1_packed_mles: Vec<Vec<Fq>>,
    pub inv_dx_c0_packed_mles: Vec<Vec<Fq>>,
    pub inv_dx_c1_packed_mles: Vec<Vec<Fq>>,

    /// Line coefficients (c0, c1, c2) ∈ Fq2 (one per step), as in the BN line function.
    pub l_c0_c0_packed_mles: Vec<Vec<Fq>>,
    pub l_c0_c1_packed_mles: Vec<Vec<Fq>>,
    pub l_c1_c0_packed_mles: Vec<Vec<Fq>>,
    pub l_c1_c1_packed_mles: Vec<Vec<Fq>>,
    pub l_c2_c0_packed_mles: Vec<Vec<Fq>>,
    pub l_c2_c1_packed_mles: Vec<Vec<Fq>>,

    /// G1 input (replicated across s,x).
    pub x_p_packed_mles: Vec<Vec<Fq>>,
    pub y_p_packed_mles: Vec<Vec<Fq>>,

    /// Per-step operand Q (for add rows; arbitrary/zero on double rows), replicated across x.
    pub x_q_c0_packed_mles: Vec<Vec<Fq>>,
    pub x_q_c1_packed_mles: Vec<Vec<Fq>>,
    pub y_q_c0_packed_mles: Vec<Vec<Fq>>,
    pub y_q_c1_packed_mles: Vec<Vec<Fq>>,

    /// Step selectors (replicated across x).
    pub is_double_packed_mles: Vec<Vec<Fq>>,
    pub is_add_packed_mles: Vec<Vec<Fq>>,

    /// Line evaluation as an Fq12 element, expanded to 16 MLE evals per step and packed.
    pub l_val_packed_mles: Vec<Vec<Fq>>,

    /// g(x) for ring-switching, replicated across step dimension (one per pair).
    pub g_packed_mles: Vec<Vec<Fq>>,

    /// Selector polynomials for mapping sparse-034 Fq12 coefficients to MLE evals, replicated across step.
    pub selector_0_packed_mles: Vec<Vec<Fq>>,
    pub selector_1_packed_mles: Vec<Vec<Fq>>,
    pub selector_2_packed_mles: Vec<Vec<Fq>>,
    pub selector_3_packed_mles: Vec<Vec<Fq>>,
    pub selector_4_packed_mles: Vec<Vec<Fq>>,
    pub selector_5_packed_mles: Vec<Vec<Fq>>,

    /// Number of active steps in the trace (<= 128).
    pub num_steps: usize,
}

impl MultiMillerLoopSteps {
    pub fn new(g1s: &[G1Affine], g2s: &[G2Affine]) -> Self {
        assert_eq!(
            g1s.len(),
            g2s.len(),
            "MultiMillerLoopSteps::new: g1s and g2s must have same length"
        );
        let num_pairs = g1s.len();

        let mut result = Fq12::one();

        let mut f_packed_mles = Vec::with_capacity(num_pairs);
        let mut f_next_packed_mles = Vec::with_capacity(num_pairs);
        let mut quotient_packed_mles = Vec::with_capacity(num_pairs);

        let mut t_x_c0_packed_mles = Vec::with_capacity(num_pairs);
        let mut t_x_c1_packed_mles = Vec::with_capacity(num_pairs);
        let mut t_y_c0_packed_mles = Vec::with_capacity(num_pairs);
        let mut t_y_c1_packed_mles = Vec::with_capacity(num_pairs);
        let mut t_x_c0_next_packed_mles = Vec::with_capacity(num_pairs);
        let mut t_x_c1_next_packed_mles = Vec::with_capacity(num_pairs);
        let mut t_y_c0_next_packed_mles = Vec::with_capacity(num_pairs);
        let mut t_y_c1_next_packed_mles = Vec::with_capacity(num_pairs);

        let mut lambda_c0_packed_mles = Vec::with_capacity(num_pairs);
        let mut lambda_c1_packed_mles = Vec::with_capacity(num_pairs);
        let mut inv_dx_c0_packed_mles = Vec::with_capacity(num_pairs);
        let mut inv_dx_c1_packed_mles = Vec::with_capacity(num_pairs);

        let mut l_c0_c0_packed_mles = Vec::with_capacity(num_pairs);
        let mut l_c0_c1_packed_mles = Vec::with_capacity(num_pairs);
        let mut l_c1_c0_packed_mles = Vec::with_capacity(num_pairs);
        let mut l_c1_c1_packed_mles = Vec::with_capacity(num_pairs);
        let mut l_c2_c0_packed_mles = Vec::with_capacity(num_pairs);
        let mut l_c2_c1_packed_mles = Vec::with_capacity(num_pairs);

        let mut x_p_packed_mles = Vec::with_capacity(num_pairs);
        let mut y_p_packed_mles = Vec::with_capacity(num_pairs);

        let mut x_q_c0_packed_mles = Vec::with_capacity(num_pairs);
        let mut x_q_c1_packed_mles = Vec::with_capacity(num_pairs);
        let mut y_q_c0_packed_mles = Vec::with_capacity(num_pairs);
        let mut y_q_c1_packed_mles = Vec::with_capacity(num_pairs);

        let mut is_double_packed_mles = Vec::with_capacity(num_pairs);
        let mut is_add_packed_mles = Vec::with_capacity(num_pairs);

        let mut l_val_packed_mles = Vec::with_capacity(num_pairs);

        let mut g_packed_mles = Vec::with_capacity(num_pairs);
        let mut selector_0_packed_mles = Vec::with_capacity(num_pairs);
        let mut selector_1_packed_mles = Vec::with_capacity(num_pairs);
        let mut selector_2_packed_mles = Vec::with_capacity(num_pairs);
        let mut selector_3_packed_mles = Vec::with_capacity(num_pairs);
        let mut selector_4_packed_mles = Vec::with_capacity(num_pairs);
        let mut selector_5_packed_mles = Vec::with_capacity(num_pairs);

        // All traces share the same step schedule for BN254, so num_steps is identical for all pairs.
        let mut num_steps_out: usize = 0;

        // Shared polynomials for ring-switching and sparse embedding.
        let g_mle = get_g_mle();
        debug_assert_eq!(g_mle.len(), ELEM_SIZE);
        let mut g_packed = vec![Fq::zero(); PACKED_SIZE];
        for x in 0..ELEM_SIZE {
            for s in 0..STEP_SIZE {
                g_packed[x * STEP_SIZE + s] = g_mle[x];
            }
        }

        // Selector basis: coefficients at sparse-034 slots (0,3,4) in Fq2, split into (c0,c1).
        let basis0_c0 = fq12_sparse_034(Fq2::new(Fq::one(), Fq::zero()), Fq2::zero(), Fq2::zero());
        let basis0_c1 = fq12_sparse_034(Fq2::new(Fq::zero(), Fq::one()), Fq2::zero(), Fq2::zero());
        let basis3_c0 = fq12_sparse_034(Fq2::zero(), Fq2::new(Fq::one(), Fq::zero()), Fq2::zero());
        let basis3_c1 = fq12_sparse_034(Fq2::zero(), Fq2::new(Fq::zero(), Fq::one()), Fq2::zero());
        let basis4_c0 = fq12_sparse_034(Fq2::zero(), Fq2::zero(), Fq2::new(Fq::one(), Fq::zero()));
        let basis4_c1 = fq12_sparse_034(Fq2::zero(), Fq2::zero(), Fq2::new(Fq::zero(), Fq::one()));

        let sel0_4 = fq12_to_multilinear_evals(&basis0_c0);
        let sel1_4 = fq12_to_multilinear_evals(&basis0_c1);
        let sel2_4 = fq12_to_multilinear_evals(&basis3_c0);
        let sel3_4 = fq12_to_multilinear_evals(&basis3_c1);
        let sel4_4 = fq12_to_multilinear_evals(&basis4_c0);
        let sel5_4 = fq12_to_multilinear_evals(&basis4_c1);

        debug_assert_eq!(sel0_4.len(), ELEM_SIZE);
        debug_assert_eq!(sel1_4.len(), ELEM_SIZE);
        debug_assert_eq!(sel2_4.len(), ELEM_SIZE);
        debug_assert_eq!(sel3_4.len(), ELEM_SIZE);
        debug_assert_eq!(sel4_4.len(), ELEM_SIZE);
        debug_assert_eq!(sel5_4.len(), ELEM_SIZE);

        let sel_pack = |sel_4: &[Fq]| -> Vec<Fq> {
            let mut packed = vec![Fq::zero(); PACKED_SIZE];
            for x in 0..ELEM_SIZE {
                for s in 0..STEP_SIZE {
                    packed[x * STEP_SIZE + s] = sel_4[x];
                }
            }
            packed
        };
        let sel0_packed = sel_pack(&sel0_4);
        let sel1_packed = sel_pack(&sel1_4);
        let sel2_packed = sel_pack(&sel2_4);
        let sel3_packed = sel_pack(&sel3_4);
        let sel4_packed = sel_pack(&sel4_4);
        let sel5_packed = sel_pack(&sel5_4);

        for (pair_idx, (&p, &q)) in g1s.iter().zip(g2s.iter()).enumerate() {
            let trace = trace_single_pair(p, q);
            if pair_idx == 0 {
                num_steps_out = trace.num_steps;
            } else {
                debug_assert_eq!(
                    num_steps_out, trace.num_steps,
                    "expected all Miller loop traces to have same step count"
                );
            }

            result *= trace.final_f;

            f_packed_mles.push(trace.f_packed);
            f_next_packed_mles.push(trace.f_next_packed);
            quotient_packed_mles.push(trace.quotient_packed);

            t_x_c0_packed_mles.push(trace.t_x_c0_packed);
            t_x_c1_packed_mles.push(trace.t_x_c1_packed);
            t_y_c0_packed_mles.push(trace.t_y_c0_packed);
            t_y_c1_packed_mles.push(trace.t_y_c1_packed);
            t_x_c0_next_packed_mles.push(trace.t_x_c0_next_packed);
            t_x_c1_next_packed_mles.push(trace.t_x_c1_next_packed);
            t_y_c0_next_packed_mles.push(trace.t_y_c0_next_packed);
            t_y_c1_next_packed_mles.push(trace.t_y_c1_next_packed);

            lambda_c0_packed_mles.push(trace.lambda_c0_packed);
            lambda_c1_packed_mles.push(trace.lambda_c1_packed);
            inv_dx_c0_packed_mles.push(trace.inv_dx_c0_packed);
            inv_dx_c1_packed_mles.push(trace.inv_dx_c1_packed);

            l_c0_c0_packed_mles.push(trace.l0_c0_packed);
            l_c0_c1_packed_mles.push(trace.l0_c1_packed);
            l_c1_c0_packed_mles.push(trace.l1_c0_packed);
            l_c1_c1_packed_mles.push(trace.l1_c1_packed);
            l_c2_c0_packed_mles.push(trace.l2_c0_packed);
            l_c2_c1_packed_mles.push(trace.l2_c1_packed);

            x_p_packed_mles.push(trace.x_p_packed);
            y_p_packed_mles.push(trace.y_p_packed);

            x_q_c0_packed_mles.push(trace.x_q_c0_packed);
            x_q_c1_packed_mles.push(trace.x_q_c1_packed);
            y_q_c0_packed_mles.push(trace.y_q_c0_packed);
            y_q_c1_packed_mles.push(trace.y_q_c1_packed);

            is_double_packed_mles.push(trace.is_double_packed);
            is_add_packed_mles.push(trace.is_add_packed);

            l_val_packed_mles.push(trace.l_val_packed);

            // Replicate shared polynomials per pair (for the current Stage1 interface).
            g_packed_mles.push(g_packed.clone());
            selector_0_packed_mles.push(sel0_packed.clone());
            selector_1_packed_mles.push(sel1_packed.clone());
            selector_2_packed_mles.push(sel2_packed.clone());
            selector_3_packed_mles.push(sel3_packed.clone());
            selector_4_packed_mles.push(sel4_packed.clone());
            selector_5_packed_mles.push(sel5_packed.clone());
        }

        Self {
            g1_points: g1s.to_vec(),
            g2_points: g2s.to_vec(),
            result,

            f_packed_mles,
            f_next_packed_mles,
            quotient_packed_mles,

            t_x_c0_packed_mles,
            t_x_c1_packed_mles,
            t_y_c0_packed_mles,
            t_y_c1_packed_mles,
            t_x_c0_next_packed_mles,
            t_x_c1_next_packed_mles,
            t_y_c0_next_packed_mles,
            t_y_c1_next_packed_mles,

            lambda_c0_packed_mles,
            lambda_c1_packed_mles,
            inv_dx_c0_packed_mles,
            inv_dx_c1_packed_mles,

            l_c0_c0_packed_mles,
            l_c0_c1_packed_mles,
            l_c1_c0_packed_mles,
            l_c1_c1_packed_mles,
            l_c2_c0_packed_mles,
            l_c2_c1_packed_mles,

            x_p_packed_mles,
            y_p_packed_mles,
            x_q_c0_packed_mles,
            x_q_c1_packed_mles,
            y_q_c0_packed_mles,
            y_q_c1_packed_mles,
            is_double_packed_mles,
            is_add_packed_mles,
            l_val_packed_mles,
            g_packed_mles,
            selector_0_packed_mles,
            selector_1_packed_mles,
            selector_2_packed_mles,
            selector_3_packed_mles,
            selector_4_packed_mles,
            selector_5_packed_mles,

            num_steps: num_steps_out,
        }
    }
}

struct SinglePairTrace {
    num_steps: usize,
    final_f: Fq12,

    f_packed: Vec<Fq>,
    f_next_packed: Vec<Fq>,
    quotient_packed: Vec<Fq>,

    t_x_c0_packed: Vec<Fq>,
    t_x_c1_packed: Vec<Fq>,
    t_y_c0_packed: Vec<Fq>,
    t_y_c1_packed: Vec<Fq>,
    t_x_c0_next_packed: Vec<Fq>,
    t_x_c1_next_packed: Vec<Fq>,
    t_y_c0_next_packed: Vec<Fq>,
    t_y_c1_next_packed: Vec<Fq>,

    lambda_c0_packed: Vec<Fq>,
    lambda_c1_packed: Vec<Fq>,
    inv_dx_c0_packed: Vec<Fq>,
    inv_dx_c1_packed: Vec<Fq>,

    l0_c0_packed: Vec<Fq>,
    l0_c1_packed: Vec<Fq>,
    l1_c0_packed: Vec<Fq>,
    l1_c1_packed: Vec<Fq>,
    l2_c0_packed: Vec<Fq>,
    l2_c1_packed: Vec<Fq>,

    x_p_packed: Vec<Fq>,
    y_p_packed: Vec<Fq>,

    x_q_c0_packed: Vec<Fq>,
    x_q_c1_packed: Vec<Fq>,
    y_q_c0_packed: Vec<Fq>,
    y_q_c1_packed: Vec<Fq>,

    is_double_packed: Vec<Fq>,
    is_add_packed: Vec<Fq>,

    l_val_packed: Vec<Fq>,
}

fn g2_mul_by_char(mut q: G2Affine) -> G2Affine {
    // Multiply by field characteristic, matching ark-ec BN implementation.
    // See `ark-ec/src/models/bn/g2.rs::mul_by_char`.
    if q.infinity {
        return q;
    }
    q.x.frobenius_map_in_place(1);
    q.x *= &Bn254Config::TWIST_MUL_BY_Q_X;
    q.y.frobenius_map_in_place(1);
    q.y *= &Bn254Config::TWIST_MUL_BY_Q_Y;
    q
}

fn trace_single_pair(p: G1Affine, q: G2Affine) -> SinglePairTrace {
    assert!(!p.is_zero(), "G1 point at infinity not supported yet");
    assert!(!q.is_zero(), "G2 point at infinity not supported yet");

    let g_mle = get_g_mle();

    let one = Fq::one();
    let zero = Fq::zero();
    let two = Fq::from(2u64);
    let three = Fq::from(3u64);

    let mut f = Fq12::one();
    let mut t = q;

    // Per-step witnesses (step-major).
    let mut f_step_mles: Vec<Vec<Fq>> = Vec::new();
    let mut f_next_step_mles: Vec<Vec<Fq>> = Vec::new();
    let mut quotient_step_mles: Vec<Vec<Fq>> = Vec::new();
    let mut l_val_step_mles: Vec<Vec<Fq>> = Vec::new();

    let mut t_x: Vec<Fq2> = Vec::new();
    let mut t_y: Vec<Fq2> = Vec::new();
    let mut t_x_next: Vec<Fq2> = Vec::new();
    let mut t_y_next: Vec<Fq2> = Vec::new();

    let mut lambda: Vec<Fq2> = Vec::new();
    let mut inv_dx: Vec<Fq2> = Vec::new();

    let mut l0: Vec<Fq2> = Vec::new();
    let mut l1: Vec<Fq2> = Vec::new();
    let mut l2: Vec<Fq2> = Vec::new();

    let mut x_q_steps: Vec<Fq2> = Vec::new();
    let mut y_q_steps: Vec<Fq2> = Vec::new();

    let mut is_double_steps: Vec<Fq> = Vec::new();
    let mut is_add_steps: Vec<Fq> = Vec::new();

    // Final two additions use Frobenius points.
    let q1 = g2_mul_by_char(q);
    let mut q2 = g2_mul_by_char(q1);
    q2.y = -q2.y;

    let neg_q = G2Affine {
        x: q.x,
        y: -q.y,
        infinity: q.infinity,
    };

    // Build the operation schedule (double always, then optional add, then final adds with q1/q2).
    let mut op_points: Vec<Option<G2Affine>> = Vec::new();
    let mut op_is_double: Vec<bool> = Vec::new();

    for bit in Bn254Config::ATE_LOOP_COUNT.iter().rev().skip(1) {
        op_points.push(None);
        op_is_double.push(true);

        match *bit {
            1 => {
                op_points.push(Some(q));
                op_is_double.push(false);
            }
            -1 => {
                op_points.push(Some(neg_q));
                op_is_double.push(false);
            }
            _ => {}
        }
    }

    // Two final additions.
    op_points.push(Some(q1));
    op_is_double.push(false);
    op_points.push(Some(q2));
    op_is_double.push(false);

    let num_steps = op_is_double.len();
    // We keep one extra "terminal state" row in the packed trace so that `*_next` columns can be
    // shift-checked against the corresponding `*` columns (i.e. `*_next(s) = *(s+1)`), matching
    // the packed GT exp pattern. This requires room for `num_steps + 1`.
    assert!(
        num_steps + 1 <= STEP_SIZE,
        "miller-loop trace too long for 7 step vars with terminal row: num_steps+1={} > {STEP_SIZE}",
        num_steps + 1
    );

    for (is_dbl, op_q_opt) in op_is_double.into_iter().zip(op_points.into_iter()) {
        // Record current T.
        t_x.push(t.x);
        t_y.push(t.y);

        if is_dbl {
            is_double_steps.push(one);
            is_add_steps.push(zero);

            // lambda = 3*x^2 / (2*y)
            let mut num = t.x.square();
            num.mul_assign_by_fp(&three);
            let mut den = t.y;
            den.mul_assign_by_fp(&two); // 2y
            let den_inv = den
                .inverse()
                .expect("2y must be invertible in doubling step");
            let lam = num * den_inv;
            lambda.push(lam);
            inv_dx.push(Fq2::zero());

            // T_next = 2T in affine.
            let x3 = lam.square() - t.x - t.x;
            let y3 = lam * (t.x - x3) - t.y;
            let t_next = G2Affine {
                x: x3,
                y: y3,
                infinity: false,
            };

            // Line coefficients (c0,c1,c2) ∈ Fq2 for the tangent line, unscaled:
            //   c0 = -2y
            //   c1 = 3x^2
            //   c2 = 2y^2 - 3x^3  (chosen so the line vanishes at T)
            let mut c0 = t.y;
            c0.mul_assign_by_fp(&two);
            c0 = -c0;

            let mut c1 = t.x.square();
            c1.mul_assign_by_fp(&three);

            let mut c2_left = t.y.square();
            c2_left.mul_assign_by_fp(&two);

            let mut c2_right = t.x.square() * t.x;
            c2_right.mul_assign_by_fp(&three);

            let c2 = c2_left - c2_right;

            l0.push(c0);
            l1.push(c1);
            l2.push(c2);

            // For double rows, operand Q is unused; fill with 0.
            x_q_steps.push(Fq2::zero());
            y_q_steps.push(Fq2::zero());

            // Build line element as sparse 034 and update f.
            let mut c0_scaled = c0;
            c0_scaled.mul_assign_by_fp(&p.y);
            let mut c3_scaled = c1;
            c3_scaled.mul_assign_by_fp(&p.x);
            let c4 = c2;

            let line = fq12_sparse_034(c0_scaled, c3_scaled, c4);

            let mut f_next = f;
            f_next.square_in_place();
            f_next *= line;

            // MLE expansions.
            let f_mle = fq12_to_multilinear_evals(&f);
            let f_next_mle = fq12_to_multilinear_evals(&f_next);
            let l_mle = fq12_to_multilinear_evals(&line);

            // Quotient MLE for ring-switching.
            let mut q_mle = vec![Fq::zero(); ELEM_SIZE];
            for x in 0..ELEM_SIZE {
                let a = f_mle[x].square(); // f^2 at this cube point
                let b = l_mle[x];
                let c = f_next_mle[x];
                let num = a * b - c;
                if !g_mle[x].is_zero() {
                    q_mle[x] = num / g_mle[x];
                }
            }

            f_step_mles.push(f_mle);
            f_next_step_mles.push(f_next_mle);
            quotient_step_mles.push(q_mle);
            l_val_step_mles.push(l_mle);

            t_x_next.push(t_next.x);
            t_y_next.push(t_next.y);

            f = f_next;
            t = t_next;
        } else {
            is_double_steps.push(zero);
            is_add_steps.push(one);

            let op_q = op_q_opt.expect("add step must have operand point");
            x_q_steps.push(op_q.x);
            y_q_steps.push(op_q.y);

            // lambda = (y_q - y_t) / (x_q - x_t)
            let dx = op_q.x - t.x;
            let dy = op_q.y - t.y;
            let inv = dx.inverse().expect("dx must be invertible in add step");
            let lam = dy * inv;
            lambda.push(lam);
            inv_dx.push(inv);

            // T_next = T + Q in affine.
            let x3 = lam.square() - t.x - op_q.x;
            let y3 = lam * (t.x - x3) - t.y;
            let t_next = G2Affine {
                x: x3,
                y: y3,
                infinity: false,
            };

            // Line coefficients (c0,c1,c2) ∈ Fq2 for the chord line, unscaled:
            //   c0 = x_t - x_q
            //   c1 = y_q - y_t
            //   c2 = x_q*y_t - x_t*y_q
            let c0 = t.x - op_q.x;
            let c1 = op_q.y - t.y;
            let c2 = op_q.x * t.y - t.x * op_q.y;

            l0.push(c0);
            l1.push(c1);
            l2.push(c2);

            // Build line element as sparse 034 and update f.
            let mut c0_scaled = c0;
            c0_scaled.mul_assign_by_fp(&p.y);
            let mut c3_scaled = c1;
            c3_scaled.mul_assign_by_fp(&p.x);
            let c4 = c2;

            let line = fq12_sparse_034(c0_scaled, c3_scaled, c4);

            let f_next = f * line;

            // MLE expansions.
            let f_mle = fq12_to_multilinear_evals(&f);
            let f_next_mle = fq12_to_multilinear_evals(&f_next);
            let l_mle = fq12_to_multilinear_evals(&line);

            // Quotient MLE for ring-switching.
            let mut q_mle = vec![Fq::zero(); ELEM_SIZE];
            for x in 0..ELEM_SIZE {
                let a = f_mle[x]; // f at this cube point
                let b = l_mle[x];
                let c = f_next_mle[x];
                let num = a * b - c;
                if !g_mle[x].is_zero() {
                    q_mle[x] = num / g_mle[x];
                }
            }

            f_step_mles.push(f_mle);
            f_next_step_mles.push(f_next_mle);
            quotient_step_mles.push(q_mle);
            l_val_step_mles.push(l_mle);

            t_x_next.push(t_next.x);
            t_y_next.push(t_next.y);

            f = f_next;
            t = t_next;
        }
    }

    // Append a terminal state row so that:
    // - `f(s,x)` includes the final accumulator value `f_S` at s=num_steps, and
    // - `T(s)` includes the final G2 state at s=num_steps,
    // enabling shift sumchecks like `f_next(s,x) = f(s+1,x)` and `T_next(s) = T(s+1)`.
    //
    // This terminal row is treated as inactive by setting `is_double=is_add=0`; the per-step
    // MultiMillerLoop constraints are gated by `is_active` and therefore do not constrain it.
    let terminal_f_mle = fq12_to_multilinear_evals(&f);
    f_step_mles.push(terminal_f_mle);
    f_next_step_mles.push(vec![Fq::zero(); ELEM_SIZE]);
    quotient_step_mles.push(vec![Fq::zero(); ELEM_SIZE]);
    l_val_step_mles.push(vec![Fq::zero(); ELEM_SIZE]);

    t_x.push(t.x);
    t_y.push(t.y);
    t_x_next.push(Fq2::zero());
    t_y_next.push(Fq2::zero());

    lambda.push(Fq2::zero());
    inv_dx.push(Fq2::zero());
    l0.push(Fq2::zero());
    l1.push(Fq2::zero());
    l2.push(Fq2::zero());
    x_q_steps.push(Fq2::zero());
    y_q_steps.push(Fq2::zero());
    is_double_steps.push(zero);
    is_add_steps.push(zero);

    // Pack step × element tables.
    let num_packed_steps = num_steps + 1; // include terminal row
    let f_packed = pack_step_and_elem(&f_step_mles, num_packed_steps);
    let f_next_packed = pack_step_and_elem(&f_next_step_mles, num_packed_steps);
    let quotient_packed = pack_step_and_elem(&quotient_step_mles, num_packed_steps);
    let l_val_packed = pack_step_and_elem(&l_val_step_mles, num_packed_steps);

    // Pack step-only columns (replicated across x).
    let t_x_c0_packed = pack_step_only_fq2_c0(&t_x, num_packed_steps);
    let t_x_c1_packed = pack_step_only_fq2_c1(&t_x, num_packed_steps);
    let t_y_c0_packed = pack_step_only_fq2_c0(&t_y, num_packed_steps);
    let t_y_c1_packed = pack_step_only_fq2_c1(&t_y, num_packed_steps);

    let t_x_c0_next_packed = pack_step_only_fq2_c0(&t_x_next, num_packed_steps);
    let t_x_c1_next_packed = pack_step_only_fq2_c1(&t_x_next, num_packed_steps);
    let t_y_c0_next_packed = pack_step_only_fq2_c0(&t_y_next, num_packed_steps);
    let t_y_c1_next_packed = pack_step_only_fq2_c1(&t_y_next, num_packed_steps);

    let lambda_c0_packed = pack_step_only_fq2_c0(&lambda, num_packed_steps);
    let lambda_c1_packed = pack_step_only_fq2_c1(&lambda, num_packed_steps);
    let inv_dx_c0_packed = pack_step_only_fq2_c0(&inv_dx, num_packed_steps);
    let inv_dx_c1_packed = pack_step_only_fq2_c1(&inv_dx, num_packed_steps);

    let l0_c0_packed = pack_step_only_fq2_c0(&l0, num_packed_steps);
    let l0_c1_packed = pack_step_only_fq2_c1(&l0, num_packed_steps);
    let l1_c0_packed = pack_step_only_fq2_c0(&l1, num_packed_steps);
    let l1_c1_packed = pack_step_only_fq2_c1(&l1, num_packed_steps);
    let l2_c0_packed = pack_step_only_fq2_c0(&l2, num_packed_steps);
    let l2_c1_packed = pack_step_only_fq2_c1(&l2, num_packed_steps);

    let x_p_vals = vec![p.x; num_packed_steps];
    let y_p_vals = vec![p.y; num_packed_steps];
    let x_p_packed = pack_step_only(&x_p_vals, num_packed_steps);
    let y_p_packed = pack_step_only(&y_p_vals, num_packed_steps);

    let x_q_c0: Vec<Fq> = x_q_steps.iter().map(|v| v.c0).collect();
    let x_q_c1: Vec<Fq> = x_q_steps.iter().map(|v| v.c1).collect();
    let y_q_c0: Vec<Fq> = y_q_steps.iter().map(|v| v.c0).collect();
    let y_q_c1: Vec<Fq> = y_q_steps.iter().map(|v| v.c1).collect();
    let x_q_c0_packed = pack_step_only(&x_q_c0, num_packed_steps);
    let x_q_c1_packed = pack_step_only(&x_q_c1, num_packed_steps);
    let y_q_c0_packed = pack_step_only(&y_q_c0, num_packed_steps);
    let y_q_c1_packed = pack_step_only(&y_q_c1, num_packed_steps);

    let is_double_packed = pack_step_only(&is_double_steps, num_packed_steps);
    let is_add_packed = pack_step_only(&is_add_steps, num_packed_steps);

    SinglePairTrace {
        num_steps,
        final_f: f,

        f_packed,
        f_next_packed,
        quotient_packed,

        t_x_c0_packed,
        t_x_c1_packed,
        t_y_c0_packed,
        t_y_c1_packed,
        t_x_c0_next_packed,
        t_x_c1_next_packed,
        t_y_c0_next_packed,
        t_y_c1_next_packed,

        lambda_c0_packed,
        lambda_c1_packed,
        inv_dx_c0_packed,
        inv_dx_c1_packed,

        l0_c0_packed,
        l0_c1_packed,
        l1_c0_packed,
        l1_c1_packed,
        l2_c0_packed,
        l2_c1_packed,

        x_p_packed,
        y_p_packed,

        x_q_c0_packed,
        x_q_c1_packed,
        y_q_c0_packed,
        y_q_c1_packed,

        is_double_packed,
        is_add_packed,

        l_val_packed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Bn254;
    use ark_ec::pairing::Pairing;

    #[test]
    fn test_single_pair_miller_loop_final_exponent_matches_pairing() {
        // Use the standard generators.
        let p = G1Affine::generator();
        let q = G2Affine::generator();

        let steps = MultiMillerLoopSteps::new(&[p], &[q]);
        assert_eq!(steps.f_packed_mles[0].len(), PACKED_SIZE);
        assert_eq!(steps.f_next_packed_mles[0].len(), PACKED_SIZE);
        assert_eq!(steps.quotient_packed_mles[0].len(), PACKED_SIZE);
        assert!(steps.num_steps > 0);

        // Our witness returns the Miller loop output (before final exponentiation).
        let our_f = steps.result;

        // Compare after final exponentiation against arkworks pairing.
        let expected = Bn254::pairing(p, q).0;
        let got = Bn254::final_exponentiation(ark_ec::pairing::MillerLoopOutput(our_f))
            .unwrap()
            .0;
        assert_eq!(got, expected);
    }
}
