// Small Value Optimization (SVO) helpers for Spartan first sum-check

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::r1cs::builder::{Constraint, OffsetEqConstraint, eval_offset_lc};
use ark_ff::Zero;
use rayon::prelude::*;

pub mod svo_helpers {
    use super::*;

    // SVOEvalPoint enum definition
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum SVOEvalPoint {
        Zero,
        One,
        Infinity,
    }

    // Function to evaluate Az, Bz for a given R1CS row with fully binary SVO prefix
    #[inline]
    pub fn evaluate_Az_Bz_for_r1cs_row_binary<F: JoltField>(
        uniform_constraints: &[Constraint],
        cross_step_constraints: &[OffsetEqConstraint],
        flattened_polynomials: &[&MultilinearPolynomial<F>],
        current_step_idx: usize,
        constraint_idx_within_step: usize, // Full constraint index including SVO bits (all binary)
        num_uniform_constraints: usize,
        num_total_steps: usize,
    ) -> (i128, i128) {
        // Returns (az_coeff_i128, bz_coeff_i128)
        let mut az_i128 = 0i128;
        let mut bz_i128 = 0i128;

        if constraint_idx_within_step < num_uniform_constraints {
            let constraint = &uniform_constraints[constraint_idx_within_step];
            if !constraint.a.terms().is_empty() {
                az_i128 = constraint
                    .a
                    .evaluate_row(flattened_polynomials, current_step_idx);
            }
            if !az_i128.is_zero() && !constraint.b.terms().is_empty() {
                bz_i128 = constraint
                    .b
                    .evaluate_row(flattened_polynomials, current_step_idx);
            }
        } else if constraint_idx_within_step < num_uniform_constraints + cross_step_constraints.len() {
            let cross_step_constraint_idx = constraint_idx_within_step - num_uniform_constraints;
            let constraint = &cross_step_constraints[cross_step_constraint_idx];
            let next_step_index_opt = if current_step_idx + 1 < num_total_steps {
                Some(current_step_idx + 1)
            } else {
                None
            };

            let eq_a_eval = eval_offset_lc(
                &constraint.a,
                flattened_polynomials,
                current_step_idx,
                next_step_index_opt,
            );
            let eq_b_eval = eval_offset_lc(
                &constraint.b,
                flattened_polynomials,
                current_step_idx,
                next_step_index_opt,
            );
            let temp_az = eq_a_eval - eq_b_eval;

            if !temp_az.is_zero() {
                az_i128 = temp_az;
                // Optional: Assert cond is zero:
                // let cond_eval = eval_offset_lc_i64(&constraint.cond, flattened_polynomials, current_step_idx, next_step_index_opt);
                // assert_eq!(cond_eval, 0, "Cross-step constraint violated");
            } else {
                bz_i128 = eval_offset_lc(
                    &constraint.cond,
                    flattened_polynomials,
                    current_step_idx,
                    next_step_index_opt,
                );
            }
        }
        (az_i128, bz_i128)
    }

    /// Special case when `num_svo_rounds == 1`
    /// In this case, we know that there is 3 - 2 = 1 temp_tA accumulator,
    /// corresponding to temp_tA(infty), which should be updated via
    /// temp_tA[infty] += e_in_val * (az[1] - az[0]) * (bz[1] - bz[0])
    #[inline]
    pub fn compute_and_update_tA_inplace_1<F: JoltField>(
        binary_az_evals: &[i128],
        binary_bz_evals: &[i128],
        e_in_val: &F,
        temp_tA: &mut [F],
    ) {
        assert!(binary_az_evals.len() == 2);
        assert!(binary_bz_evals.len() == 2);
        assert!(temp_tA.len() == 1);
        let az_I = binary_az_evals[1] - binary_az_evals[0];
        if az_I != 0 {
            let bz_I = binary_bz_evals[1] - binary_bz_evals[0];
            let product_i128 = az_I.checked_mul(bz_I).expect("Az_I*Bz_I product overflow i128");
            temp_tA[0] += e_in_val.mul_i128(product_i128);
        }
    }

    /// Special case when `num_svo_rounds == 2`
    /// In this case, we know that there are 4 binary evals of Az and Bz,
    /// corresponding to (0,0) => 0, (0,1) => 1, (1,0) => 2, (1,1) => 3. (i.e. big endian, TODO: double-check this)
    /// 
    /// There are 9 - 4 = 5 temp_tA accumulators, with the following logic (in this order 0 => 4 indexing):
    /// temp_tA[0,∞] += e_in_val * (az[0,1] - az[0,0]) * (bz[0,1] - bz[0,0])
    /// temp_tA[1,∞] += e_in_val * (az[1,1] - az[1,0]) * (bz[1,1] - bz[1,0])
    /// temp_tA[∞,0] += e_in_val * (az[1,0] - az[0,0]) * (bz[1,0] - bz[0,0])
    /// temp_tA[∞,1] += e_in_val * (az[1,1] - az[0,1]) * (bz[1,1] - bz[0,1])
    /// temp_tA[∞,∞] += e_in_val * (az[1,∞] - az[0,∞]) * (bz[1,∞] - bz[0,∞])
    #[inline]
    pub fn compute_and_update_tA_inplace_2<F: JoltField>(
        binary_az_evals: &[i128],
        binary_bz_evals: &[i128],
        e_in_val: &F,
        temp_tA: &mut [F],
    ) {
        assert!(binary_az_evals.len() == 4);
        assert!(binary_bz_evals.len() == 4);
        assert!(temp_tA.len() == 5);
        // Binary evaluations: (Y0,Y1) -> index Y0*2 + Y1
        let az00 = binary_az_evals[0]; let bz00 = binary_bz_evals[0]; // Y0=0, Y1=0
        let az01 = binary_az_evals[1]; let bz01 = binary_bz_evals[1]; // Y0=0, Y1=1
        let az10 = binary_az_evals[2]; let bz10 = binary_bz_evals[2]; // Y0=1, Y1=0
        let az11 = binary_az_evals[3]; let bz11 = binary_bz_evals[3]; // Y0=1, Y1=1

        // Extended evaluations (points with at least one 'I')
        // temp_tA indices follow the order: (0,I), (1,I), (I,0), (I,1), (I,I)

        // 1. Point (0,I) -> temp_tA[0]
        let az_0I = az01 - az00;
        let bz_0I = bz01 - bz00;
        if az_0I != 0 { // Due to constraint layout, it's more likely for Az to be zero
            let prod_0I = az_0I.checked_mul(bz_0I).expect("Product overflow for (0,I)");
            temp_tA[0] += e_in_val.mul_i128(prod_0I);
        }

        // 2. Point (1,I) -> temp_tA[1]
        let az_1I = az11 - az10;
        let bz_1I = bz11 - bz10;
        if az_1I != 0 {
            let prod_1I = az_1I.checked_mul(bz_1I).expect("Product overflow for (1,I)");
            temp_tA[1] += e_in_val.mul_i128(prod_1I);
        }

        // 3. Point (I,0) -> temp_tA[2]
        let az_I0 = az10 - az00;
        if az_I0 != 0 {
            let bz_I0 = bz10 - bz00;
            let prod_I0 = az_I0.checked_mul(bz_I0).expect("Product overflow for (I,0)");
            temp_tA[2] += e_in_val.mul_i128(prod_I0);
        }

        // 4. Point (I,1) -> temp_tA[3]
        let az_I1 = az11 - az01;
        if az_I1 != 0 {
            let bz_I1 = bz11 - bz01;
            let prod_I1 = az_I1.checked_mul(bz_I1).expect("Product overflow for (I,1)");
            temp_tA[3] += e_in_val.mul_i128(prod_I1);
        }

        // 5. Point (I,I) -> temp_tA[4]
        // az_II = az(1,I) - az(0,I) = az_1I - az_0I
        // bz_II = bz(1,I) - bz(0,I) = bz_1I - bz_0I
        let az_II = az_1I - az_0I;
        if az_II != 0 {
            let bz_II = bz_1I - bz_0I;
            let prod_II = az_II.checked_mul(bz_II).expect("Product overflow for (I,I)");
            temp_tA[4] += e_in_val.mul_i128(prod_II);
        }
    }

    /// Special case when `num_svo_rounds == 3`
    /// In this case, we know that there are 8 binary evals of Az and Bz,
    /// corresponding to (0,0,0) => 0, (0,0,1) => 1, (0,1,0) => 2, (0,1,1) => 3, (1,0,0) => 4,
    /// (1,0,1) => 5, (1,1,0) => 6, (1,1,1) => 7. (i.e. big endian, TODO: double-check this)
    /// 
    /// There are 27 - 8 = 19 temp_tA accumulators, with the following logic:
    /// temp_tA[0,0,∞] += e_in_val * (az[0,0,1] - az[0,0,0]) * (bz[0,0,1] - bz[0,0,0])
    /// (todo: fill the rest)
    #[inline]
    pub fn compute_and_update_tA_inplace_3<F: JoltField>(
        binary_az_evals: &[i128],
        binary_bz_evals: &[i128],
        e_in_val: &F,
        temp_tA: &mut [F],
    ) {
        assert!(binary_az_evals.len() == 8);
        assert!(binary_bz_evals.len() == 8);
        assert!(temp_tA.len() == 19);

        if e_in_val.is_zero() {
            return;
        }

        // Binary evaluations: (Y0,Y1,Y2) -> index Y0*4 + Y1*2 + Y2
        let az000 = binary_az_evals[0]; let bz000 = binary_bz_evals[0];
        let az001 = binary_az_evals[1]; let bz001 = binary_bz_evals[1];
        let az010 = binary_az_evals[2]; let bz010 = binary_bz_evals[2];
        let az011 = binary_az_evals[3]; let bz011 = binary_bz_evals[3];
        let az100 = binary_az_evals[4]; let bz100 = binary_bz_evals[4];
        let az101 = binary_az_evals[5]; let bz101 = binary_bz_evals[5];
        let az110 = binary_az_evals[6]; let bz110 = binary_bz_evals[6];
        let az111 = binary_az_evals[7]; let bz111 = binary_bz_evals[7];

        // --- FIRST GROUP: I at END (first occurrence) ---
        
        // 1. Point (0,0,I) -> temp_tA[0]
        let az_00I = az001 - az000;
        let bz_00I = bz001 - bz000;
        if az_00I != 0 {
            let prod = az_00I.checked_mul(bz_00I).expect("Product overflow for (0,0,I)");
            temp_tA[0] += e_in_val.mul_i128(prod);
        }

        // 2. Point (0,1,I) -> temp_tA[1]
        let az_01I = az011 - az010;
        let bz_01I = bz011 - bz010;
        if az_01I != 0 {
            let prod = az_01I.checked_mul(bz_01I).expect("Product overflow for (0,1,I)");
            temp_tA[1] += e_in_val.mul_i128(prod);
        }

        // 3. Point (1,0,I) -> temp_tA[5]
        let az_10I = az101 - az100;
        let bz_10I = bz101 - bz100;
        if az_10I != 0 {
            let prod = az_10I.checked_mul(bz_10I).expect("Product overflow for (1,0,I)");
            temp_tA[5] += e_in_val.mul_i128(prod);
        }

        // 4. Point (1,1,I) -> temp_tA[6]
        let az_11I = az111 - az110;
        let bz_11I = bz111 - bz110;
        if az_11I != 0 {
            let prod = az_11I.checked_mul(bz_11I).expect("Product overflow for (1,1,I)");
            temp_tA[6] += e_in_val.mul_i128(prod);
        }

        // --- SECOND GROUP: I in MIDDLE (first occurrence) ---

        // 5. Point (0,I,0) -> temp_tA[2]
        let az_0I0 = az010 - az000;
        let bz_0I0 = bz010 - bz000;
        if az_0I0 != 0 {
            let prod = az_0I0.checked_mul(bz_0I0).expect("Product overflow for (0,I,0)");
            temp_tA[2] += e_in_val.mul_i128(prod);
        }

        // 6. Point (0,I,1) -> temp_tA[3]
        let az_0I1 = az011 - az001;
        let bz_0I1 = bz011 - bz001;
        if az_0I1 != 0 {
            let prod = az_0I1.checked_mul(bz_0I1).expect("Product overflow for (0,I,1)");
            temp_tA[3] += e_in_val.mul_i128(prod);
        }

        // 7. Point (1,I,0) -> temp_tA[7]
        let az_1I0 = az110 - az100;
        let bz_1I0 = bz110 - bz100;
        if az_1I0 != 0 {
            let prod = az_1I0.checked_mul(bz_1I0).expect("Product overflow for (1,I,0)");
            temp_tA[7] += e_in_val.mul_i128(prod);
        }

        // 8. Point (1,I,1) -> temp_tA[8]
        let az_1I1 = az111 - az101;
        let bz_1I1 = bz111 - bz101;
        if az_1I1 != 0 {
            let prod = az_1I1.checked_mul(bz_1I1).expect("Product overflow for (1,I,1)");
            temp_tA[8] += e_in_val.mul_i128(prod);
        }

        // --- THIRD GROUP: I at BEGINNING (first occurrence) ---

        // 9. Point (I,0,0) -> temp_tA[10]
        let az_I00 = az100 - az000;
        let bz_I00 = bz100 - bz000;
        if az_I00 != 0 {
            let prod = az_I00.checked_mul(bz_I00).expect("Product overflow for (I,0,0)");
            temp_tA[10] += e_in_val.mul_i128(prod);
        }

        // 10. Point (I,0,1) -> temp_tA[11]
        let az_I01 = az101 - az001;
        let bz_I01 = bz101 - bz001;
        if az_I01 != 0 {
            let prod = az_I01.checked_mul(bz_I01).expect("Product overflow for (I,0,1)");
            temp_tA[11] += e_in_val.mul_i128(prod);
        }

        // 11. Point (I,1,0) -> temp_tA[13]
        let az_I10 = az110 - az010;
        let bz_I10 = bz110 - bz010;
        if az_I10 != 0 {
            let prod = az_I10.checked_mul(bz_I10).expect("Product overflow for (I,1,0)");
            temp_tA[13] += e_in_val.mul_i128(prod);
        }

        // 12. Point (I,1,1) -> temp_tA[14]
        let az_I11 = az111 - az011;
        let bz_I11 = bz111 - bz011;
        if az_I11 != 0 {
            let prod = az_I11.checked_mul(bz_I11).expect("Product overflow for (I,1,1)");
            temp_tA[14] += e_in_val.mul_i128(prod);
        }

        // --- FOURTH GROUP: MULTIPLE I's ---
        
        // 13. Point (0,I,I) -> temp_tA[4]
        let az_0II = az_01I - az_00I;
        let bz_0II = bz_01I - bz_00I;
        if az_0II != 0 {
            let prod = az_0II.checked_mul(bz_0II).expect("Product overflow for (0,I,I)");
            temp_tA[4] += e_in_val.mul_i128(prod);
        }

        // 14. Point (1,I,I) -> temp_tA[9]
        let az_1II = az_11I - az_10I;
        let bz_1II = bz_11I - bz_10I;
        if az_1II != 0 {
            let prod = az_1II.checked_mul(bz_1II).expect("Product overflow for (1,I,I)");
            temp_tA[9] += e_in_val.mul_i128(prod);
        }

        // 15. Point (I,0,I) -> temp_tA[12]
        let az_I0I = az_I01 - az_I00;
        let bz_I0I = bz_I01 - bz_I00;
        if az_I0I != 0 {
            let prod = az_I0I.checked_mul(bz_I0I).expect("Product overflow for (I,0,I)");
            temp_tA[12] += e_in_val.mul_i128(prod);
        }

        // 16. Point (I,1,I) -> temp_tA[15]
        let az_I1I = az_I11 - az_I10;
        let bz_I1I = bz_I11 - bz_I10;
        if az_I1I != 0 {
            let prod = az_I1I.checked_mul(bz_I1I).expect("Product overflow for (I,1,I)");
            temp_tA[15] += e_in_val.mul_i128(prod);
        }

        // 17. Point (I,I,0) -> temp_tA[16]
        let az_II0 = az_1I0 - az_0I0;
        let bz_II0 = bz_1I0 - bz_0I0;
        if az_II0 != 0 {
            let prod = az_II0.checked_mul(bz_II0).expect("Product overflow for (I,I,0)");
            temp_tA[16] += e_in_val.mul_i128(prod);
        }

        // 18. Point (I,I,1) -> temp_tA[17]
        let az_II1 = az_1I1 - az_0I1;
        let bz_II1 = bz_1I1 - bz_0I1;
        if az_II1 != 0 {
            let prod = az_II1.checked_mul(bz_II1).expect("Product overflow for (I,I,1)");
            temp_tA[17] += e_in_val.mul_i128(prod);
        }

        // 19. Point (I,I,I) -> temp_tA[18]
        let az_III = az_1II - az_0II;
        let bz_III = bz_1II - bz_0II;
        if az_III != 0 {
            let prod = az_III.checked_mul(bz_III).expect("Product overflow for (I,I,I)");
            temp_tA[18] += e_in_val.mul_i128(prod);
        }
    }

    // Do the same for 2, 3

    /// Performs in-place multilinear extension on ternary evaluation vectors
    /// and updates the temporary accumulator `temp_tA` with the product contribution.
    /// This is the generic version
    #[inline]
    pub fn compute_and_update_tA_inplace<F: JoltField>(
        ternary_az_evals: &mut [i128], // Size 3^l0, holds Az evals. Initially populated at binary points. Modified in-place.
        ternary_bz_evals: &mut [i128], // Size 3^l0, holds Bz evals. Initially populated at binary points. Modified in-place.
        num_svo_rounds: usize,        // l_0
        e_in_val: &F,                  // E_in[x_in] factor for the current x_in
        temp_tA: &mut [F],            // Accumulator vector (size 3^l0), updated with E_in * P_ext contribution.
    ) {
        let num_ternary_points = ternary_az_evals.len();
        let expected_ternary_points = 3_usize.checked_pow(num_svo_rounds as u32)
                                        .expect("Ternary points count overflow");
        debug_assert_eq!(num_ternary_points, expected_ternary_points);
        debug_assert_eq!(ternary_bz_evals.len(), num_ternary_points);
        debug_assert_eq!(temp_tA.len(), num_ternary_points);

        if num_svo_rounds == 0 {
            if num_ternary_points > 0 { // Should be size 1 if l0=0
                let az0 = ternary_az_evals[0];
                let bz0 = ternary_bz_evals[0];
                if az0 != 0 && bz0 != 0 { // Early check
                    let product_i128 = az0.checked_mul(bz0).expect("Az0*Bz0 product overflow i128 in SVO base case");
                    temp_tA[0] += e_in_val.mul_i128(product_i128);
                }
            }
            return;
        }

        // --- In-place Extension Phase ---
        for j_dim_idx in 0..num_svo_rounds {
            let base: usize = 3;
            let num_prefix_vars = j_dim_idx;
            let num_suffix_vars = num_svo_rounds - 1 - j_dim_idx;
            let num_prefix_points = base.checked_pow(num_prefix_vars as u32).expect("Prefix power overflow");
            let num_suffix_points = base.checked_pow(num_suffix_vars as u32).expect("Suffix power overflow");

            for prefix_int in 0..num_prefix_points {
                for suffix_int in 0..num_suffix_points {
                    let mut coords = vec![0usize; num_svo_rounds];
                    let mut current_suffix_int = suffix_int;
                    for i in (j_dim_idx + 1..num_svo_rounds).rev() {
                        coords[i] = current_suffix_int % base;
                        current_suffix_int /= base;
                    }
                    let mut current_prefix_int = prefix_int;
                    for i in (0..j_dim_idx).rev() {
                        coords[i] = current_prefix_int % base;
                        current_prefix_int /= base;
                    }

                    coords[j_dim_idx] = 0;
                    let idx0 = svo_helpers::get_fixed_radix_index(&coords, base, num_svo_rounds);
                    coords[j_dim_idx] = 1;
                    let idx1 = svo_helpers::get_fixed_radix_index(&coords, base, num_svo_rounds);
                    coords[j_dim_idx] = 2;
                    let idx_inf = svo_helpers::get_fixed_radix_index(&coords, base, num_svo_rounds);

                    let az_p0 = ternary_az_evals[idx0];
                    let az_p1 = ternary_az_evals[idx1];
                    let bz_p0 = ternary_bz_evals[idx0];
                    let bz_p1 = ternary_bz_evals[idx1];

                    let az_p_inf = az_p1.saturating_sub(az_p0);
                    let bz_p_inf = bz_p1.saturating_sub(bz_p0);

                    ternary_az_evals[idx_inf] = az_p_inf;
                    ternary_bz_evals[idx_inf] = bz_p_inf;
                }
            }
        } // End loop over j_dim_idx

        // --- Accumulation Phase ---
        // After the j_dim_idx loops, ternary_az_evals and ternary_bz_evals 
        // contain the fully extended evaluations.
        temp_tA
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, tA_val)| {
                let az_final = ternary_az_evals[idx];
                let bz_final = ternary_bz_evals[idx];

                // Premature breaking condition
                if az_final == 0 || bz_final == 0 {
                    return; // In parallel context, this exits the current closure iteration
                }

                let product_p_i128 = az_final.checked_mul(bz_final).expect("Az_ext*Bz_ext product overflow i128");
                // product_p_i128 cannot be zero here due to the check above

                // Accumulate contribution from this x_in using mul_i128
                *tA_val += e_in_val.mul_i128(product_p_i128);
            });
    }

    /// Hardcoded version for `num_svo_rounds == 1`
    #[inline]
    pub fn distribute_tA_to_svo_accumulators_1<F: JoltField>(
        task_tA_accumulator_vec: &[F], 
        x_out_val: usize,             
        E_out_vec: &[Vec<F>],
        task_svo_accs: &mut Vec<(Vec<F>, Vec<F>)>,
    ) {
        assert!(task_tA_accumulator_vec.len() == 1);
        assert!(E_out_vec.len() >= 1);
        assert!(task_svo_accs.len() >= 1);
        
        // For NUM_SVO_ROUNDS=1, we have only one non-binary point (Y0=I)
        // This maps to only one round (s=0) with only one v-configuration (v=[])
        
        // Get task_tA value for the Infinity point
        let tA_for_I = task_tA_accumulator_vec[0];
        
        // For this point (Y0=I), we have:
        // round_s = 0, v_config = [], u = Infinity, y_suffix = []
        let E_out_0_val = get_E_out_s_val::<F>(
            &E_out_vec[0],          // E_out for round 0
            0,                      // num_y_suffix_vars = 0
            0,                      // y_suffix_as_int = 0
            x_out_val,
        );
        
        // Update accumulator for round 0, v_config=[], u=Infinity
        // v_config_idx = 0 (empty v_config)
        task_svo_accs[0].1[0] += E_out_0_val * tA_for_I;
    }

    /// Hardcoded version for `num_svo_rounds == 2`
    #[inline]
    pub fn distribute_tA_to_svo_accumulators_2<F: JoltField>(
        task_tA_accumulator_vec: &[F], 
        x_out_val: usize,             
        E_out_vec: &[Vec<F>],
        task_svo_accs: &mut Vec<(Vec<F>, Vec<F>)>,
    ) {
        assert!(task_tA_accumulator_vec.len() == 5);
        assert!(E_out_vec.len() >= 2);
        assert!(task_svo_accs.len() >= 2);
        
        // For NUM_SVO_ROUNDS=2, we have 5 non-binary points with their corresponding mappings:
        // 0: (0,I) -> s=0, v=[], u=I, y=[0] (temp_tA[0])
        // 1: (1,I) -> s=0, v=[], u=I, y=[1] (temp_tA[1])
        // 2: (I,0) -> s=0, v=[], u=I, y=[] and s=1, v=[I], u=0, y=[] (temp_tA[2])
        // 3: (I,1) -> s=0, v=[], u=I, y=[] and s=1, v=[I], u=1, y=[] (temp_tA[3])
        // 4: (I,I) -> s=0, v=[], u=I, y=[] and s=1, v=[I], u=I, y=[] (temp_tA[4])

        // Points 0 and 1: (0,I) and (1,I) both map to round_s=0, v_config=[], u=I
        // Point 0: (0,I) -> round_s=0, v_config=[], u=I, y=[0]
        let tA_0I = task_tA_accumulator_vec[0];
        let E_out_0_val = get_E_out_s_val::<F>(
            &E_out_vec[0],        // E_out for round 0
            1,                    // num_y_suffix_vars = 1
            0,                    // y_suffix_as_int = 0
            x_out_val,
        );
        // v_config_idx = 0 (empty v_config)
        task_svo_accs[0].1[0] += E_out_0_val * tA_0I;

        // Point 1: (1,I) -> round_s=0, v_config=[], u=I, y=[1]
        let tA_1I = task_tA_accumulator_vec[1];
        let E_out_0_val = get_E_out_s_val::<F>(
            &E_out_vec[0],        // E_out for round 0
            1,                    // num_y_suffix_vars = 1
            1,                    // y_suffix_as_int = 1
            x_out_val,
        );
        
        // v_config_idx = 0 (empty v_config)
        task_svo_accs[0].1[0] += E_out_0_val * tA_1I;

        // Points 2,3,4: (I,0), (I,1), (I,I) all map to round_s=1, v_config=[I]
        // Point 2: (I,0) -> round_s=1, v_config=[I], u=0, y=[]
        let tA_I0 = task_tA_accumulator_vec[2];
        let E_out_1_val = get_E_out_s_val::<F>(
            &E_out_vec[1],        // E_out for round 1
            0,                    // num_y_suffix_vars = 0
            0,                    // y_suffix_as_int = 0
            x_out_val,
        );
        
        // v_config_idx = 2 (for v_config=[I])
        task_svo_accs[1].0[2] += E_out_1_val * tA_I0;

        // Point 3: (I,1) -> round_s=1, v_config=[I], u=1, y=[]
        let tA_I1 = task_tA_accumulator_vec[3];
        let E_out_1_val = get_E_out_s_val::<F>(
            &E_out_vec[1],        // E_out for round 1
            0,                    // num_y_suffix_vars = 0
            0,                    // y_suffix_as_int = 0
            x_out_val,
        );
        
        // v_config_idx = 2 (for v_config=[I])
        task_svo_accs[1].0[2] += E_out_1_val * tA_I1;

        // Point 4: (I,I) -> round_s=1, v_config=[I], u=I, y=[]
        let tA_II = task_tA_accumulator_vec[4];
        let E_out_1_val = get_E_out_s_val::<F>(
            &E_out_vec[1],        // E_out for round 1
            0,                    // num_y_suffix_vars = 0
            0,                    // y_suffix_as_int = 0
            x_out_val,
        );
        
        // v_config_idx = 2 (for v_config=[I])
        task_svo_accs[1].1[2] += E_out_1_val * tA_II;
    }

    /// Hardcoded version for `num_svo_rounds == 3`
    #[inline]
    pub fn distribute_tA_to_svo_accumulators_3<F: JoltField>(
        task_tA_accumulator_vec: &[F], 
        x_out_val: usize,             
        E_out_vec: &[Vec<F>],
        task_svo_accs: &mut Vec<(Vec<F>, Vec<F>)>,
    ) {
        assert!(task_tA_accumulator_vec.len() == 19);
        assert!(E_out_vec.len() >= 3);
        assert!(task_svo_accs.len() >= 3);
        
        // For NUM_SVO_ROUNDS=3, we have 19 non-binary points with various mappings
        // This is a simplified implementation - in a real scenario, we would handle all 19 points
        
        // Just a placeholder for now - to be expanded
        for idx in 0..19 {
            let tA_val = task_tA_accumulator_vec[idx];
            
            // Map the temp_tA index to appropriate round, v_config, and u
            // This is a simplified version - in practice we'd have 19 specific cases
            if idx < 4 {
                // First few points contribute to round 0
                let E_out_0_val = get_E_out_s_val::<F>(
                    &E_out_vec[0],
                    0,
                    0,
                    x_out_val,
                );
                
                task_svo_accs[0].1[0] += E_out_0_val * tA_val;
            }
        }
    }

    // Distributes the accumulated tA values (sum over x_in) for a single x_out_val 
    // to the appropriate SVO round accumulators.
    #[inline]
    pub fn distribute_tA_to_svo_accumulators<F: JoltField>(
        task_tA_accumulator_vec: &[F], 
        x_out_val: usize,             
        num_svo_rounds: usize,
        num_ternary_points: usize,
        E_out_vec: &[Vec<F>],         
        all_idx_mapping_results: &[Vec<(usize, Vec<SVOEvalPoint>, SVOEvalPoint, usize, usize)>], // Updated tuple
        task_svo_accs: &mut Vec<(Vec<F>, Vec<F>)>,
    ) {
        if num_svo_rounds == 0 {
            return; 
        }

        for beta_idx in 0..num_ternary_points {
            let tA_for_this_beta = task_tA_accumulator_vec[beta_idx];
            if tA_for_this_beta.is_zero() { continue; }

            let relevant_mappings = &all_idx_mapping_results[beta_idx];

            for (round_s, v_config_vec, u_eval_point, y_suffix_as_int, num_y_suffix_vars) in relevant_mappings.iter().cloned() {
                let mut is_v_config_binary = true;
                for v_comp in &v_config_vec {
                    if *v_comp == SVOEvalPoint::Infinity {
                        is_v_config_binary = false;
                        break;
                    }
                }
                if is_v_config_binary && u_eval_point == SVOEvalPoint::Zero {
                    continue;
                }
                
                let E_out_s_val = get_E_out_s_val::<F>(
                    &E_out_vec[round_s],
                    num_y_suffix_vars, // Use precomputed num_y_suffix_vars
                    y_suffix_as_int,   // Use precomputed y_suffix_as_int
                    x_out_val, 
                );
                let v_config_idx = map_v_config_to_idx(&v_config_vec, round_s);

                if !E_out_s_val.is_zero() {
                    match u_eval_point {
                        SVOEvalPoint::Zero => {
                            task_svo_accs[round_s].0[v_config_idx] += E_out_s_val * tA_for_this_beta;
                        }
                        SVOEvalPoint::Infinity => {
                            task_svo_accs[round_s].1[v_config_idx] += E_out_s_val * tA_for_this_beta;
                        }
                        SVOEvalPoint::One => { panic!("SVO accumulators only store u=0/inf"); }
                    }
                }
            }
        }
    }

    // Computes Teq factor for E_out_s
    // This function now uses the precomputed E_out_s evaluations from eq_poly.E1.
    // Corresponds to accessing E_out,i[y, x_out] in Algorithm 6 (proc:precompute-algo-6, Line 14).
    #[inline]
    pub fn get_E_out_s_val<F: JoltField>(
        E_out_s_evals: &[F],              // Precomputed evaluations E_out,i for round i=s
        num_y_suffix_vars: usize,         // Number of variables in y_suffix (length of y in E_out,i[y, x_out])
        y_suffix_as_int: usize,           // Integer representation of y_suffix (LSB)
        x_out_val: usize,                 // Integer assignment for x_out variables (MSB)
    ) -> F {
        // y_suffix_as_int is LSB, x_out_val is MSB. Combine to form the index into E_out,i.
        let combined_idx = (x_out_val << num_y_suffix_vars) | y_suffix_as_int;
        E_out_s_evals[combined_idx] // E_out,i[y, x_out]
    }

    // Implements Definition 4 ("idx4") from the paper to map an extended SVO prefix
    // to (round_s, v_config, u_eval, y_suffix_as_int, num_y_suffix_vars) tuples.
    // Corresponds to Algorithm 6 (proc:precompute-algo-6, Line 13): determining the target indices (i, v, u, y) from beta.
    #[inline]
    pub fn idx_mapping(
        svo_prefix_extended: &[SVOEvalPoint], 
        num_svo_rounds: usize,
    ) -> Vec<(usize, Vec<SVOEvalPoint>, SVOEvalPoint, usize, usize)> { // Added y_suffix_as_int, num_y_suffix_vars
        let mut result = Vec::new();
        if num_svo_rounds == 0 {
            return result;
        }
        assert_eq!(svo_prefix_extended.len(), num_svo_rounds);

        for s in 0..num_svo_rounds { 
            let v_config: Vec<SVOEvalPoint> = svo_prefix_extended[0..s].to_vec();
            let u_eval = svo_prefix_extended[s]; 

            if !(u_eval == SVOEvalPoint::Zero || u_eval == SVOEvalPoint::Infinity) {
                continue;
            }

            let y_suffix_components = &svo_prefix_extended[s + 1..num_svo_rounds];
            let mut y_suffix_is_binary = true;
            let mut y_suffix_as_int = 0;
            for (bit_idx, &point) in y_suffix_components.iter().enumerate() {
                if point == SVOEvalPoint::Infinity { 
                    y_suffix_is_binary = false;
                    break;
                }
                if point == SVOEvalPoint::One {
                    y_suffix_as_int |= 1 << bit_idx; // LSB is y_{s+1}
                }
            }
            let num_y_suffix_vars = y_suffix_components.len();

            if !y_suffix_is_binary {
                continue;
            }

            result.push((s, v_config, u_eval, y_suffix_as_int, num_y_suffix_vars));
        }
        result
    }

    // Maps a v-configuration (Vec<SVOEvalPoint> of length s) to a unique usize index (0 to 3^s - 1).
    // Used for indexing into the SVO accumulator vectors.
    #[inline]
    pub fn map_v_config_to_idx(v_config: &[SVOEvalPoint], _round_s: usize) -> usize {
        // Example: Treat SVOEvalPoint::{Zero, One, Infinity} as digits 0, 1, 2 in base 3.
        let mut index = 0;
        let base: usize = 3;
        for (i, point) in v_config.iter().enumerate() {
            let digit = match point {
                SVOEvalPoint::Zero => 0,
                SVOEvalPoint::One => 1,
                SVOEvalPoint::Infinity => 2,
            };
            index += digit * base.pow(i as u32);
        }
        index
    }

    // Helper to compute the integer index for a point represented by its coordinate values
    /// in a fixed-base radix system.
    /// Assumes coordinates and the single `base` apply to all dimensions.
    /// Coordinates are MSB-first (index 0 is highest order variable Y_0).
    /// Calculates index assuming LSB-first contribution (Y_0 has lowest stride).
    #[inline]
    pub fn get_fixed_radix_index(point_coords: &[usize], base: usize, num_vars: usize) -> usize {
        debug_assert_eq!(num_vars, point_coords.len());
        let mut index = 0;
        let mut stride = 1;
        // Iterate from LSB (Y_{l0-1}) to MSB (Y_0)
        for i in (0..num_vars).rev() {
            debug_assert!(point_coords[i] < base, "Coord {} out of bounds for base {}", point_coords[i], base);
            index += point_coords[i] * stride;
            if i > 0 { // Avoid overflow on last stride calculation
               // Use checked_mul for safety
               stride = stride.checked_mul(base).expect("Base power overflow computing stride");
            }
        }
        index
    }

    /// Helper to convert a binary index (0..2^l0-1) to its equivalent index
    /// in a base-3 system where binary 0 -> ternary 0, binary 1 -> ternary 1.
    /// Binary index is interpreted LSB first (bit 0 corresponds to Y_0).
    /// Ternary index is calculated LSB first (Y_0 has stride 1).
    #[inline]
    pub fn binary_to_ternary_index(binary_idx: usize, num_svo_rounds: usize) -> usize {
        let mut ternary_idx = 0;
        let mut current_binary_val = binary_idx;
        let mut stride = 1;
        for i in 0..num_svo_rounds {
            let bit = current_binary_val % 2; // Get LSB
            ternary_idx += bit * stride; // Add 0 or 1 * stride
            // Stride calculation needs to be careful for the *next* iteration
            if i < num_svo_rounds - 1 {
                 stride = stride.checked_mul(3).expect("Stride overflow in binary_to_ternary_index");
            }
            current_binary_val /= 2;
        }
        if current_binary_val != 0 {
            // This indicates the binary_idx was too large for num_svo_rounds
            panic!("binary_idx {} too large for {} SVO rounds", binary_idx, num_svo_rounds);
        }
        ternary_idx
    }

    /// Precomputes the mapping from binary indices to their ternary equivalents.
    #[inline]
    pub fn precompute_binary_to_ternary_indices(num_svo_rounds: usize) -> Vec<usize> {
        if num_svo_rounds == 0 {
            // If l0=0, there's one point (empty prefix), index 0 maps to 0.
            return vec![0];
        }
        let num_binary_points = 1 << num_svo_rounds;
        (0..num_binary_points)
            .map(|bin_idx| binary_to_ternary_index(bin_idx, num_svo_rounds))
            .collect()
    }

    // Helper to convert an integer index (base 3) to its SVOEvalPoint vector representation.
    #[inline]
    pub fn get_svo_prefix_extended_from_idx(beta_idx: usize, num_svo_rounds: usize) -> Vec<SVOEvalPoint> {
        if num_svo_rounds == 0 {
            return vec![];
        }
        let mut temp_beta_coords = vec![0usize; num_svo_rounds];
        let mut temp_beta_int = beta_idx;
        for i in (0..num_svo_rounds).rev() { // MSB first for coords array from LSB of int
            temp_beta_coords[i] = temp_beta_int % 3;
            temp_beta_int /= 3;
        }
        temp_beta_coords.iter().map(|&coord| match coord {
            0 => SVOEvalPoint::Zero,
            1 => SVOEvalPoint::One,
            2 => SVOEvalPoint::Infinity,
            _ => unreachable!("Invalid coordinate in beta_idx to SVOEvalPoint conversion"),
        }).collect::<Vec<_>>()
    }

    // Precomputes idx_mapping results for all possible beta values.
    #[inline]
    pub fn precompute_all_idx_mappings(
        num_svo_rounds: usize,
        num_ternary_points: usize,
    ) -> Vec<Vec<(usize, Vec<SVOEvalPoint>, SVOEvalPoint, usize, usize)>> { // Updated tuple
        if num_svo_rounds == 0 {
            return vec![vec![]]; 
        }
        (0..num_ternary_points).map(|beta_idx| {
            let svo_prefix = get_svo_prefix_extended_from_idx(beta_idx, num_svo_rounds);
            idx_mapping(&svo_prefix, num_svo_rounds) // idx_mapping now returns the richer tuple
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::svo_helpers::*;
    use SVOEvalPoint::*;
    use crate::field::JoltField;
    use ark_bn254::Fr as TestField;
    use ark_ff::{Zero, One};

    /// Tests the `get_svo_prefix_extended_from_idx` function.
    /// This function converts a base-3 integer index into a vector of `SVOEvalPoint`s.
    /// It covers cases for 0, 1, 2, and 3 SVO rounds, ensuring correct MSB-first
    /// conversion from the ternary representation of the index.
    #[test]
    fn test_get_svo_prefix_extended_from_idx() {
        // num_svo_rounds = 0
        assert_eq!(get_svo_prefix_extended_from_idx(0, 0), vec![]);

        // num_svo_rounds = 1
        assert_eq!(get_svo_prefix_extended_from_idx(0, 1), vec![SVOEvalPoint::Zero]);
        assert_eq!(get_svo_prefix_extended_from_idx(1, 1), vec![SVOEvalPoint::One]);
        assert_eq!(get_svo_prefix_extended_from_idx(2, 1), vec![SVOEvalPoint::Infinity]);

        // num_svo_rounds = 2. Indices are in base 3, MSB first in output vector.
        // 0 = 00_3
        assert_eq!(get_svo_prefix_extended_from_idx(0, 2), vec![SVOEvalPoint::Zero, SVOEvalPoint::Zero]);
        // 1 = 01_3
        assert_eq!(get_svo_prefix_extended_from_idx(1, 2), vec![SVOEvalPoint::Zero, SVOEvalPoint::One]);
        // 2 = 02_3
        assert_eq!(get_svo_prefix_extended_from_idx(2, 2), vec![SVOEvalPoint::Zero, SVOEvalPoint::Infinity]);
        // 3 = 10_3
        assert_eq!(get_svo_prefix_extended_from_idx(3, 2), vec![SVOEvalPoint::One, SVOEvalPoint::Zero]);
        // 4 = 11_3
        assert_eq!(get_svo_prefix_extended_from_idx(4, 2), vec![SVOEvalPoint::One, SVOEvalPoint::One]);
        // 8 = 22_3
        assert_eq!(get_svo_prefix_extended_from_idx(8, 2), vec![SVOEvalPoint::Infinity, SVOEvalPoint::Infinity]);

        // num_svo_rounds = 3
        // 0 = 000_3
        assert_eq!(get_svo_prefix_extended_from_idx(0, 3), vec![SVOEvalPoint::Zero, SVOEvalPoint::Zero, SVOEvalPoint::Zero]);
        // 13 = 111_3  (1*9 + 1*3 + 1*1)
        assert_eq!(get_svo_prefix_extended_from_idx(13, 3), vec![SVOEvalPoint::One, SVOEvalPoint::One, SVOEvalPoint::One]);
        // 26 = 222_3 (2*9 + 2*3 + 2*1)
        assert_eq!(get_svo_prefix_extended_from_idx(26, 3), vec![SVOEvalPoint::Infinity, SVOEvalPoint::Infinity, SVOEvalPoint::Infinity]);
         // 5 = 012_3 (0*9 + 1*3 + 2*1)
        assert_eq!(get_svo_prefix_extended_from_idx(5, 3), vec![SVOEvalPoint::Zero, SVOEvalPoint::One, SVOEvalPoint::Infinity]);
    }

    /// Tests the `map_v_config_to_idx` function.
    /// This function converts a v-configuration (a slice of `SVOEvalPoint`s representing
    /// a prefix of an SVO evaluation point) into a unique base-3 integer index.
    /// The `_round_s` parameter is currently unused by the function but kept for API consistency.
    /// It tests with empty, single-element, and multi-element v-configurations.
    /// The v-config elements are treated as LSB-first digits for the base-3 number.
    #[test]
    fn test_map_v_config_to_idx() {
        assert_eq!(map_v_config_to_idx(&[], 0), 0);
        assert_eq!(map_v_config_to_idx(&[SVOEvalPoint::Zero], 0), 0);
        assert_eq!(map_v_config_to_idx(&[SVOEvalPoint::One], 0), 1);
        assert_eq!(map_v_config_to_idx(&[SVOEvalPoint::Infinity], 0), 2);

        // v_config is LSB first for powers of 3
        assert_eq!(map_v_config_to_idx(&[SVOEvalPoint::Zero, SVOEvalPoint::Zero], 1), 0); // 0*3^0 + 0*3^1
        assert_eq!(map_v_config_to_idx(&[SVOEvalPoint::One, SVOEvalPoint::Zero], 1), 1);  // 1*3^0 + 0*3^1
        assert_eq!(map_v_config_to_idx(&[SVOEvalPoint::Zero, SVOEvalPoint::One], 1), 3);  // 0*3^0 + 1*3^1
        assert_eq!(map_v_config_to_idx(&[SVOEvalPoint::Infinity, SVOEvalPoint::Infinity], 1), 8); // 2*3^0 + 2*3^1 = 2+6

        assert_eq!(map_v_config_to_idx(&[SVOEvalPoint::One, SVOEvalPoint::One, SVOEvalPoint::One], 2), 13); // 1*1 + 1*3 + 1*9
    }

    /// Tests the `get_fixed_radix_index` function.
    /// This function calculates a linear index from a slice of coordinates in a fixed-base radix system.
    /// It assumes coordinates are MSB-first in the input slice, but calculates the index
    /// as if contributions are LSB-first (i.e., the last coordinate has stride 1).
    /// Tests cover 0 variables, base 3, and base 2 systems with varying numbers of variables.
    #[test]
    fn test_get_fixed_radix_index() {
        // num_vars = 0
        assert_eq!(get_fixed_radix_index(&[], 3, 0), 0);
        
        // base = 3, num_vars = 1 (coords are MSB first)
        assert_eq!(get_fixed_radix_index(&[0], 3, 1), 0);
        assert_eq!(get_fixed_radix_index(&[1], 3, 1), 1);
        assert_eq!(get_fixed_radix_index(&[2], 3, 1), 2);

        // base = 3, num_vars = 2 (coords MSB first: [Y0, Y1])
        // index LSB first: Y1*stride_Y1 + Y0*stride_Y0 where stride_Y1=1, stride_Y0=base
        assert_eq!(get_fixed_radix_index(&[0, 0], 3, 2), 0); // 0*1 + 0*3
        assert_eq!(get_fixed_radix_index(&[0, 1], 3, 2), 1); // 1*1 + 0*3
        assert_eq!(get_fixed_radix_index(&[0, 2], 3, 2), 2); // 2*1 + 0*3
        assert_eq!(get_fixed_radix_index(&[1, 0], 3, 2), 3); // 0*1 + 1*3
        assert_eq!(get_fixed_radix_index(&[1, 1], 3, 2), 4); // 1*1 + 1*3
        assert_eq!(get_fixed_radix_index(&[2, 2], 3, 2), 8); // 2*1 + 2*3

        // base = 2, num_vars = 3
        assert_eq!(get_fixed_radix_index(&[0,0,0], 2, 3), 0); // 0*1+0*2+0*4
        assert_eq!(get_fixed_radix_index(&[0,0,1], 2, 3), 1); // 1*1+0*2+0*4
        assert_eq!(get_fixed_radix_index(&[0,1,0], 2, 3), 2); // 0*1+1*2+0*4
        assert_eq!(get_fixed_radix_index(&[1,1,1], 2, 3), 7); // 1*1+1*2+1*4
    }

    /// Tests the `binary_to_ternary_index` function.
    /// This function converts a binary index (representing an SVO prefix with only 0s and 1s)
    /// to its equivalent index in a base-3 system where binary 0 maps to ternary 0, and binary 1 to ternary 1.
    /// Both binary and ternary indices are interpreted LSB-first for their digits.
    /// It tests for 0, 1, 2, and 3 SVO rounds.
    #[test]
    fn test_binary_to_ternary_index() {
        assert_eq!(binary_to_ternary_index(0, 0), 0); // Special case, 0 rounds -> 1 point

        assert_eq!(binary_to_ternary_index(0, 1), 0); // 0_2 -> 0_3
        assert_eq!(binary_to_ternary_index(1, 1), 1); // 1_2 -> 1_3

        // num_svo_rounds = 2
        // Binary LSB first: Y0, Y1
        // Ternary LSB first: Y0, Y1 with base 3 strides
        assert_eq!(binary_to_ternary_index(0b00, 2), 0); // 00_3 is 0
        assert_eq!(binary_to_ternary_index(0b01, 2), 1); // 01_3 is 1
        assert_eq!(binary_to_ternary_index(0b10, 2), 3); // 10_3 is 3
        assert_eq!(binary_to_ternary_index(0b11, 2), 4); // 11_3 is 4

        // num_svo_rounds = 3
        assert_eq!(binary_to_ternary_index(0b000, 3), 0);  // 0
        assert_eq!(binary_to_ternary_index(0b001, 3), 1);  // 1
        assert_eq!(binary_to_ternary_index(0b010, 3), 3);  // 3
        assert_eq!(binary_to_ternary_index(0b011, 3), 4);  // 4
        assert_eq!(binary_to_ternary_index(0b100, 3), 9);  // 9
        assert_eq!(binary_to_ternary_index(0b101, 3), 10); // 10
        assert_eq!(binary_to_ternary_index(0b110, 3), 12); // 12
        assert_eq!(binary_to_ternary_index(0b111, 3), 13); // 13
    }

    /// Tests that `binary_to_ternary_index` panics when the binary index is too large
    /// for the specified number of SVO rounds (1 round case).
    #[test]
    #[should_panic]
    fn test_binary_to_ternary_index_panic() {
        binary_to_ternary_index(4, 1); // binary_idx 4 (100_2) too large for 1 SVO round (max 1_2)
    }
    /// Tests that `binary_to_ternary_index` panics when the binary index is too large
    /// for the specified number of SVO rounds (2 rounds case).
     #[test]
    #[should_panic]
    fn test_binary_to_ternary_index_panic_2() {
        binary_to_ternary_index(1 << 2, 2); // 2 rounds, 1<<2 = 4 is too large (max is 3)
    }


    /// Tests the `precompute_binary_to_ternary_indices` function.
    /// This function generates a lookup table mapping all possible binary indices for a given
    /// number of SVO rounds to their corresponding ternary indices (where binary 0/1 map to ternary 0/1).
    /// It tests for 0, 1, 2, and 3 SVO rounds.
    #[test]
    fn test_precompute_binary_to_ternary_indices() {
        assert_eq!(precompute_binary_to_ternary_indices(0), vec![0]);
        assert_eq!(precompute_binary_to_ternary_indices(1), vec![0, 1]);
        assert_eq!(precompute_binary_to_ternary_indices(2), vec![0, 1, 3, 4]);
        assert_eq!(precompute_binary_to_ternary_indices(3), vec![0, 1, 3, 4, 9, 10, 12, 13]);
    }

    /// Tests the `idx_mapping` function.
    /// This function implements Definition 4 ("idx4") from the Jolt paper, mapping an extended
    /// SVO prefix (beta) to a vector of tuples. Each tuple contains `(round_s, v_config, u_eval, y_suffix_as_int, num_y_suffix_vars)`
    /// relevant for SVO accumulator updates. `u_eval` must be Zero or Infinity, and the `y_suffix` must be binary.
    /// Tests cover 0, 1, 2, and 3 SVO rounds with various beta configurations.
    #[test]
    fn test_idx_mapping() {
        use SVOEvalPoint::*;
        // num_svo_rounds = 0
        assert_eq!(idx_mapping(&[], 0), vec![]);

        // num_svo_rounds = 1
        assert_eq!(idx_mapping(&[Zero], 1), vec![(0, vec![], Zero, 0, 0)]);
        assert_eq!(idx_mapping(&[One], 1), vec![]); // u_eval must be Z or I
        assert_eq!(idx_mapping(&[Infinity], 1), vec![(0, vec![], Infinity, 0, 0)]);

        // num_svo_rounds = 2
        // beta = [Z, Z]
        let res_zz = idx_mapping(&[Zero, Zero], 2);
        assert_eq!(res_zz.len(), 2);
        assert!(res_zz.contains(&(0, vec![], Zero, 0, 1))); // s=0, v=[], u=Z, y=[Z] -> y_val=0, y_len=1
        assert!(res_zz.contains(&(1, vec![Zero], Zero, 0, 0)));   // s=1, v=[Z], u=Z, y=[] -> y_val=0, y_len=0
        
        // beta = [Z, O]
        let res_zo = idx_mapping(&[Zero, One], 2);
        assert_eq!(res_zo.len(), 1);
        assert!(res_zo.contains(&(0, vec![], Zero, 1, 1))); // s=0, v=[], u=Z, y=[O] -> y_val=1, y_len=1
                                                              // s=1, v=[Z], u=O -> skipped

        // beta = [Z, I]
        let res_zi = idx_mapping(&[Zero, Infinity], 2);
        assert_eq!(res_zi.len(), 1);
                                                              // s=0, v=[], u=Z, y=[I] -> y_suffix not binary, skipped
        assert!(res_zi.contains(&(1, vec![Zero], Infinity, 0, 0))); // s=1, v=[Z], u=I, y=[] -> y_val=0, y_len=0

        // beta = [I, Z]
        let res_iz = idx_mapping(&[Infinity, Zero], 2);
        assert_eq!(res_iz.len(), 2);
        assert!(res_iz.contains(&(0, vec![], Infinity, 0, 1))); // s=0, v=[], u=I, y=[Z] -> y_val=0, y_len=1
        assert!(res_iz.contains(&(1, vec![Infinity], Zero, 0, 0))); // s=1, v=[I], u=Z, y=[] -> y_val=0, y_len=0

        // beta = [O, Z]
        let res_oz = idx_mapping(&[One, Zero], 2);
        assert_eq!(res_oz.len(), 1);
                                                            // s=0, v=[], u=O -> skipped
        assert!(res_oz.contains(&(1, vec![One], Zero, 0, 0))); // s=1, v=[O], u=Z, y=[] -> y_val=0, y_len=0
        
        // num_svo_rounds = 3
        // beta = [Z, O, I]
        let res_zoi = idx_mapping(&[Zero, One, Infinity], 3);
        assert_eq!(res_zoi.len(), 1);
        // s=0, v=[], u=Z, y=[O,I] -> y not binary, skipped
        // s=1, v=[Z], u=O -> u not Z/I, skipped
        assert!(res_zoi.contains(&(2, vec![Zero, One], Infinity, 0, 0))); // s=2, v=[Z,O], u=I, y=[]

        // beta = [Z,Z,Z]
        let res_zzz = idx_mapping(&[Zero,Zero,Zero], 3);
        assert_eq!(res_zzz.len(), 3);
        assert!(res_zzz.contains(&(0, vec![], Zero, 0, 2))); // y=[Z,Z]
        assert!(res_zzz.contains(&(1, vec![Zero], Zero, 0, 1))); // y=[Z]
        assert!(res_zzz.contains(&(2, vec![Zero,Zero], Zero, 0, 0))); // y=[]

        // beta = [I,O,Z]
        let res_ioz = idx_mapping(&[Infinity, One, Zero], 3);
        assert_eq!(res_ioz.len(), 2);
        assert!(res_ioz.contains(&(0, vec![], Infinity, 1, 2))); // y=[O,Z] (O is LSB of y_suffix components, int value becomes 1)
        // s=1, v=[I], u=O -> skip
        assert!(res_ioz.contains(&(2, vec![Infinity, One], Zero, 0, 0))); // y=[]
    }

    /// Tests the `precompute_all_idx_mappings` function.
    #[test]
    fn test_precompute_all_idx_mappings() {
        // num_svo_rounds = 0
        let map_0 = precompute_all_idx_mappings(0, 1);
        assert_eq!(map_0.len(), 1);
        assert_eq!(map_0[0], vec![]);

        // num_svo_rounds = 1
        let map_1 = precompute_all_idx_mappings(1, 3);
        assert_eq!(map_1.len(), 3);
        // beta_idx = 0 (Z): svo_prefix = [Z]
        assert_eq!(map_1[0], idx_mapping(&[Zero], 1));
        assert_eq!(map_1[0], vec![(0, vec![], Zero, 0, 0)]);
        // beta_idx = 1 (O): svo_prefix = [O]
        assert_eq!(map_1[1], idx_mapping(&[One], 1));
        assert_eq!(map_1[1], vec![]);
        // beta_idx = 2 (I): svo_prefix = [I]
        assert_eq!(map_1[2], idx_mapping(&[Infinity], 1));
        assert_eq!(map_1[2], vec![(0, vec![], Infinity, 0, 0)]);

        // num_svo_rounds = 2
        let map_2 = precompute_all_idx_mappings(2, 9);
        assert_eq!(map_2.len(), 9);
        for beta_idx in 0..9 {
            let svo_prefix = get_svo_prefix_extended_from_idx(beta_idx, 2);
            assert_eq!(map_2[beta_idx], idx_mapping(&svo_prefix, 2), 
                      "Mismatch for beta_idx {}", beta_idx);
        }
        
        // Spot check beta_idx = 0 ([Z,Z])
        let res_zz = map_2[0].clone();
        assert_eq!(res_zz.len(), 2);
        assert!(res_zz.contains(&(0, vec![], Zero, 0, 1))); 
        assert!(res_zz.contains(&(1, vec![Zero], Zero, 0, 0)));

        // Spot check beta_idx = 5 ([O,I]) = 12_3 = 1*3 + 2*1 = 5
        let svo_prefix_oi = get_svo_prefix_extended_from_idx(5, 2);
        assert_eq!(svo_prefix_oi, vec![One, Infinity]);
        let res_oi = idx_mapping(&svo_prefix_oi, 2);
        assert_eq!(map_2[5], res_oi);
        assert_eq!(res_oi.len(), 1);
        assert!(res_oi.contains(&(1, vec![One], Infinity, 0, 0)));
    }

    // Basic test for compute_and_update_tA_inplace
    #[test]
    fn test_compute_and_update_tA_inplace() {
        // Test num_svo_rounds = 0 (base case)
        let mut az_evals_0 = vec![5i128];
        let mut bz_evals_0 = vec![3i128];
        let mut temp_tA_0 = vec![TestField::zero()];
        let e_in_val_0 = TestField::from(2u64);

        compute_and_update_tA_inplace(
            &mut az_evals_0,
            &mut bz_evals_0,
            0, // num_svo_rounds
            &e_in_val_0,
            &mut temp_tA_0
        );

        // Expected: e_in_val * az * bz = 2 * 5 * 3 = 30
        assert_eq!(temp_tA_0[0], TestField::from(30u64));

        // Test num_svo_rounds = 1
        // For l0=1, we have 3 points (0,1,I) and need to test extension
        let mut az_evals_1 = vec![2i128, 4i128, 0i128]; // Only 0 and 1 points initially populated
        let mut bz_evals_1 = vec![3i128, 5i128, 0i128];
        let mut temp_tA_1 = vec![TestField::zero(); 3];
        let e_in_val_1 = TestField::from(1u64);

        compute_and_update_tA_inplace(
            &mut az_evals_1,
            &mut bz_evals_1,
            1,
            &e_in_val_1,
            &mut temp_tA_1
        );

        // After extension:
        // az_evals_1 should be [2, 4, 2] (I value is 4-2=2)
        // bz_evals_1 should be [3, 5, 2] (I value is 5-3=2)
        assert_eq!(az_evals_1, vec![2i128, 4i128, 2i128]);
        assert_eq!(bz_evals_1, vec![3i128, 5i128, 2i128]);

        // Expected: e_in_val * az * bz
        assert_eq!(temp_tA_1[0], TestField::from(6u64));  // 1 * 2 * 3 = 6
        assert_eq!(temp_tA_1[1], TestField::from(20u64)); // 1 * 4 * 5 = 20
        assert_eq!(temp_tA_1[2], TestField::from(4u64));  // 1 * 2 * 2 = 4
    }

    // Test for distribute_tA_to_svo_accumulators
    #[test]
    fn test_distribute_tA_to_svo_accumulators() {
        // Test scenario: l0=2, round_s in {0,1}
        // Num ternary points = 3^2 = 9
        let num_svo_rounds = 2;
        let num_ternary_points = 9;
        let x_out_val = 0; // Simple case with x_out=0
        #[cfg(test)]
        let iter_num_x_out_vars = 1; // Match the fixed value used in testing

        // Prepare task_tA_accumulator_vec - Only populate a few indices to test specific paths
        let mut task_tA_accumulator_vec = vec![TestField::zero(); num_ternary_points];
        // beta_idx 0 = [Z,Z] -> Should be zero according to SVO logic (fully binary)
        task_tA_accumulator_vec[0] = TestField::zero();
        // beta_idx 4 = [O,O] -> Should be zero according to SVO logic (fully binary)
        task_tA_accumulator_vec[4] = TestField::zero();
        // beta_idx 2 = [Z,I] -> Not fully binary, can be non-zero
        task_tA_accumulator_vec[2] = TestField::from(5u64);

        // Prepare E_out_vec with two vectors, one for each SVO round
        // Round 0 has num_y_suffix_vars=1, Round 1 has num_y_suffix_vars=0
        let mut E_out_vec = Vec::new();
        // For round_s=0: num_y_suffix_vars=1, num_x_out_vars=1
        // Table length 2^(1+1) = 4
        let E_out_s0 = vec![
            TestField::from(1u64), // E[0,0]
            TestField::from(2u64), // E[0,1] 
            TestField::from(3u64), // E[1,0]
            TestField::from(4u64), // E[1,1]
        ];
        // For round_s=1: num_y_suffix_vars=0, num_x_out_vars=1
        // Table length 2^(0+1) = 2
        let E_out_s1 = vec![
            TestField::from(5u64), // E[0]
            TestField::from(6u64), // E[1]
        ];
        E_out_vec.push(E_out_s0);
        E_out_vec.push(E_out_s1);

        // Precompute all the idx mappings
        let all_idx_mapping_results = precompute_all_idx_mappings(num_svo_rounds, num_ternary_points);
        
        // Initialize SVO accumulators for both rounds
        // For each round, we need two vectors: one for u=Z and one for u=I
        // Each vector has size 3^s (s=round index)
        let mut task_svo_accs = Vec::new();
        // Round 0: s=0, num_v_configs = 3^0 = 1
        task_svo_accs.push((
            vec![TestField::zero()], // u=Z, v=[] (only one config)
            vec![TestField::zero()]  // u=I, v=[] (only one config)
        ));
        // Round 1: s=1, num_v_configs = 3^1 = 3 (v can be Z, O, or I)
        task_svo_accs.push((
            vec![TestField::zero(); 3], // u=Z, v in {Z,O,I}
            vec![TestField::zero(); 3]  // u=I, v in {Z,O,I}
        ));

        // Invoke the distribution function
        distribute_tA_to_svo_accumulators(
            &task_tA_accumulator_vec,
            x_out_val,
            num_svo_rounds,
            num_ternary_points,
            &E_out_vec,
            &all_idx_mapping_results,
            &mut task_svo_accs,
        );

        // Verify results based on our knowledge of the mappings and values
        // For beta_idx = 0 ([Z,Z]), task_tA = 2:
        // - Maps to (0, [], Z, 0, 1) and (1, [Z], Z, 0, 0)
        // - For round 0: s=0, v=[], u=Z, y=0, E_out = E[0,0] = 1
        //   => task_svo_accs[0].0[0] += 1 * 2 = 2. Now expects 0 since task_tA_accumulator_vec[0] is 0.
        // - For round 1: s=1, v=[Z], u=Z, y=[], E_out = E[0] = 5
        //   => task_svo_accs[1].0[0] += 5 * 2 = 10. Now expects 0 since task_tA_accumulator_vec[0] is 0.
        assert_eq!(task_svo_accs[0].0[0], TestField::zero()); // round 0, u=Z
        assert_eq!(task_svo_accs[1].0[0], TestField::zero()); // round 1, u=Z, v=[Z]
        
        // For beta_idx = 2 ([Z,I]), task_tA = 5:
        // - Maps to (1, [Z], I, 0, 0)
        // - For round 1: s=1, v=[Z], u=I, y=[], E_out = E[0] = 5
        //   => task_svo_accs[1].1[0] += 5 * 5 = 25
        assert_eq!(task_svo_accs[1].1[0], TestField::from(25u64)); // round 1, u=I, v=[Z]
        
        // For beta_idx = 4 ([O,O]), task_tA = 3:
        // - Maps to (1, [O], O, 0, 0) - This gets skipped because u=O
        // - No contribution expected
        
        // Check that other accumulators remain at zero
        assert_eq!(task_svo_accs[0].1[0], TestField::zero()); // round 0, u=I
        assert_eq!(task_svo_accs[1].0[1], TestField::zero()); // round 1, u=Z, v=[O]
        assert_eq!(task_svo_accs[1].0[2], TestField::zero()); // round 1, u=Z, v=[I]
        assert_eq!(task_svo_accs[1].1[1], TestField::zero()); // round 1, u=I, v=[O]
        assert_eq!(task_svo_accs[1].1[2], TestField::zero()); // round 1, u=I, v=[I]
    }
}
