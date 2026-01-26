//! Regression tests for prefix-packing dense polynomial emission.

use crate::poly::dense_mlpoly::DensePolynomial;
use crate::zkvm::recursion::constraints::system::{
    ConstraintLocator, ConstraintSystem, ConstraintType, G1AddNative, G1ScalarMulNative,
    GtMulNativeRows, PolyType, RecursionMatrixShape,
};
use crate::zkvm::recursion::gt::exponentiation::{GtExpPublicInputs, GtExpWitness};
use crate::zkvm::recursion::witness_generation::emit_dense;
use ark_bn254::{Fq, Fq12, Fr};
use ark_ff::{One, Zero};

#[inline]
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut y = 0usize;
    for _ in 0..bits {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    y
}

fn shape_for(num_constraints: usize) -> RecursionMatrixShape {
    let num_constraints_padded = num_constraints.next_power_of_two();
    let num_rows_unpadded = PolyType::NUM_TYPES * num_constraints_padded;
    let num_s_vars = (num_rows_unpadded as f64).log2().ceil() as usize;
    let num_constraint_vars = 11usize;
    let num_vars = num_s_vars + num_constraint_vars;
    RecursionMatrixShape {
        num_constraints,
        num_constraints_padded,
        num_constraint_vars,
        num_s_vars,
        num_vars,
    }
}

#[test]
fn test_emit_dense_matches_bit_reversal_semantics() {
    // Build a tiny planned constraint system that exercises all native sizes:
    // - 11-var (GtExp): rho_prev + quotient
    // - 8-var  (G1ScalarMul): 8 committed polynomials
    // - 4-var  (GtMul): 4 committed polynomials
    // - 0-var  (G1Add): 13 committed polynomials

    let g_poly_4var = DensePolynomial::new(vec![Fq::zero(); 16]);

    let rho_packed: Vec<Fq> = (0..2048).map(|i| Fq::from(i as u64)).collect();
    let quotient_packed: Vec<Fq> = (0..2048).map(|i| Fq::from((10_000 + i) as u64)).collect();
    let gt_exp_witness = GtExpWitness::<Fq> {
        rho_packed,
        rho_next_packed: vec![Fq::zero(); 2048],
        quotient_packed,
        digit_lo_packed: vec![Fq::zero(); 2048],
        digit_hi_packed: vec![Fq::zero(); 2048],
        base_packed: vec![Fq::zero(); 2048],
        base2_packed: vec![Fq::zero(); 2048],
        base3_packed: vec![Fq::zero(); 2048],
        num_steps: 1,
    };

    let gt_mul_rows = GtMulNativeRows {
        lhs: (0..16).map(|i| Fq::from((200 + i) as u64)).collect(),
        rhs: (0..16).map(|i| Fq::from((300 + i) as u64)).collect(),
        result: (0..16).map(|i| Fq::from((400 + i) as u64)).collect(),
        quotient: (0..16).map(|i| Fq::from((500 + i) as u64)).collect(),
    };

    let g1_rows = G1ScalarMulNative {
        base_point: (Fq::one(), Fq::from(2u64)),
        x_a: (0..256).map(|i| Fq::from((600 + i) as u64)).collect(),
        y_a: (0..256).map(|i| Fq::from((900 + i) as u64)).collect(),
        x_t: (0..256).map(|i| Fq::from((1_200 + i) as u64)).collect(),
        y_t: (0..256).map(|i| Fq::from((1_500 + i) as u64)).collect(),
        x_a_next: (0..256).map(|i| Fq::from((1_800 + i) as u64)).collect(),
        y_a_next: (0..256).map(|i| Fq::from((2_100 + i) as u64)).collect(),
        t_indicator: (0..256).map(|i| Fq::from((2_400 + i) as u64)).collect(),
        a_indicator: (0..256).map(|i| Fq::from((2_700 + i) as u64)).collect(),
    };

    let g1_add = G1AddNative {
        x_p: Fq::from(42u64),
        y_p: Fq::from(43u64),
        ind_p: Fq::zero(),
        x_q: Fq::from(44u64),
        y_q: Fq::from(45u64),
        ind_q: Fq::zero(),
        x_r: Fq::from(46u64),
        y_r: Fq::from(47u64),
        ind_r: Fq::zero(),
        lambda: Fq::from(48u64),
        inv_delta_x: Fq::from(49u64),
        is_double: Fq::zero(),
        is_inverse: Fq::zero(),
    };

    // Constraint ordering matches the prover's plan-pass block order.
    let constraint_types = vec![
        ConstraintType::GtExp,
        ConstraintType::GtMul,
        ConstraintType::G1ScalarMul {
            base_point: g1_rows.base_point,
        },
        ConstraintType::G1Add,
    ];
    let locator_by_constraint = vec![
        ConstraintLocator::GtExp { local: 0 },
        ConstraintLocator::GtMul { local: 0 },
        ConstraintLocator::G1ScalarMul { local: 0 },
        ConstraintLocator::G1Add { local: 0 },
    ];

    let cs = ConstraintSystem {
        shape: shape_for(constraint_types.len()),
        constraint_types: constraint_types.clone(),
        locator_by_constraint,
        g_poly: g_poly_4var,
        gt_exp_witnesses: vec![gt_exp_witness],
        gt_exp_public_inputs: vec![GtExpPublicInputs::new(Fq12::one(), vec![true])],
        gt_mul_rows: vec![gt_mul_rows],
        g1_scalar_mul_rows: vec![g1_rows],
        g2_scalar_mul_rows: vec![],
        g1_add_rows: vec![g1_add],
        g2_add_rows: vec![],
        g1_scalar_mul_public_inputs: vec![
            crate::zkvm::recursion::g1::scalar_multiplication::G1ScalarMulPublicInputs::new(
                Fr::from(0u64),
            ),
        ],
        g2_scalar_mul_public_inputs: vec![],
    };

    let (dense, layout) = emit_dense(&cs);
    let evals = dense.evals_ref();

    // For every packed entry, verify the block matches bit-reversal of the native table.
    for entry in &layout.entries {
        let native_size = 1usize << entry.num_vars;
        let block = &evals[entry.offset..entry.offset + native_size];

        let loc = cs.locator_by_constraint[entry.constraint_idx];
        match entry.poly_type {
            PolyType::RhoPrev => {
                let ConstraintLocator::GtExp { local } = loc else {
                    panic!("unexpected locator for RhoPrev: {loc:?}");
                };
                let src = &cs.gt_exp_witnesses[local].rho_packed;
                for t in 0..native_size {
                    assert_eq!(block[t], src[bit_reverse(t, entry.num_vars)]);
                }
            }
            PolyType::Quotient => {
                let ConstraintLocator::GtExp { local } = loc else {
                    panic!("unexpected locator for Quotient: {loc:?}");
                };
                let src = &cs.gt_exp_witnesses[local].quotient_packed;
                for t in 0..native_size {
                    assert_eq!(block[t], src[bit_reverse(t, entry.num_vars)]);
                }
            }

            PolyType::MulLhs => {
                let ConstraintLocator::GtMul { local } = loc else {
                    panic!("unexpected locator for MulLhs: {loc:?}");
                };
                let src = &cs.gt_mul_rows[local].lhs;
                for t in 0..native_size {
                    assert_eq!(block[t], src[bit_reverse(t, entry.num_vars)]);
                }
            }
            PolyType::MulRhs => {
                let ConstraintLocator::GtMul { local } = loc else {
                    panic!("unexpected locator for MulRhs: {loc:?}");
                };
                let src = &cs.gt_mul_rows[local].rhs;
                for t in 0..native_size {
                    assert_eq!(block[t], src[bit_reverse(t, entry.num_vars)]);
                }
            }
            PolyType::MulResult => {
                let ConstraintLocator::GtMul { local } = loc else {
                    panic!("unexpected locator for MulResult: {loc:?}");
                };
                let src = &cs.gt_mul_rows[local].result;
                for t in 0..native_size {
                    assert_eq!(block[t], src[bit_reverse(t, entry.num_vars)]);
                }
            }
            PolyType::MulQuotient => {
                let ConstraintLocator::GtMul { local } = loc else {
                    panic!("unexpected locator for MulQuotient: {loc:?}");
                };
                let src = &cs.gt_mul_rows[local].quotient;
                for t in 0..native_size {
                    assert_eq!(block[t], src[bit_reverse(t, entry.num_vars)]);
                }
            }

            // Spot-check a couple of scalar-mul committed polys.
            PolyType::G1ScalarMulXA => {
                let ConstraintLocator::G1ScalarMul { local } = loc else {
                    panic!("unexpected locator for G1ScalarMulXA: {loc:?}");
                };
                let src = &cs.g1_scalar_mul_rows[local].x_a;
                for t in 0..native_size {
                    assert_eq!(block[t], src[bit_reverse(t, entry.num_vars)]);
                }
            }
            PolyType::G1ScalarMulXANext => {
                let ConstraintLocator::G1ScalarMul { local } = loc else {
                    panic!("unexpected locator for G1ScalarMulXANext: {loc:?}");
                };
                let src = &cs.g1_scalar_mul_rows[local].x_a_next;
                for t in 0..native_size {
                    assert_eq!(block[t], src[bit_reverse(t, entry.num_vars)]);
                }
            }

            // 0-var: no bit reversal effect; entry.num_vars == 0 and native_size == 1.
            PolyType::G1AddXP => {
                let ConstraintLocator::G1Add { local } = loc else {
                    panic!("unexpected locator for G1AddXP: {loc:?}");
                };
                assert_eq!(native_size, 1);
                assert_eq!(block[0], cs.g1_add_rows[local].x_p);
            }

            // Other committed entries are exercised by the loop implicitly; we don't need
            // to duplicate every mapping here.
            _ => {}
        }
    }

    // The suffix beyond unpadded_len is implicitly zero.
    for &v in &evals[layout.unpadded_len()..] {
        assert!(v.is_zero());
    }
}
