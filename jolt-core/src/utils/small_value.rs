// Small Value Optimization (SVO) helpers for Spartan first sum-check

use crate::field::{JoltField, OptimizedMulI128};

pub mod svo_helpers {
    use super::*;
    use crate::poly::split_eq_poly::GruenSplitEqPolynomial;
    use crate::poly::unipoly::CompressedUniPoly;
    use crate::subprotocols::sumcheck::process_eq_sumcheck_round;
    use crate::utils::transcript::Transcript;

    // Import constants for assertions and USES_SMALL_VALUE_OPTIMIZATION
    use crate::poly::spartan_interleaved_poly::{
        num_accums_eval_zero, num_accums_eval_infty,
    };
    use crate::r1cs::spartan::small_value_optimization::USES_SMALL_VALUE_OPTIMIZATION;

    // SVOEvalPoint enum definition
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum SVOEvalPoint {
        Zero,
        One,
        Infinity,
    }

    // Moved helper functions from sumcheck.rs
    #[inline]
    const fn three_pow(k: usize) -> usize {
        3_usize.checked_pow(k as u32).expect("3^k overflow")
    }

    #[inline]
    const fn two_pow(k: usize) -> usize {
        2_usize.checked_pow(k as u32).expect("2^k overflow")
    }

    #[inline]
    fn get_v_config_digits(mut k_global: usize, num_vars: usize) -> Vec<usize> {
        if num_vars == 0 {
            return vec![];
        }
        let mut digits = vec![0; num_vars];
        let mut i = num_vars;
        while i > 0 { // Fill from LSB of digits vec, which corresponds to LSB of k_global
            digits[i - 1] = k_global % 3;
            k_global /= 3;
            i -= 1;
        }
        digits
    }

    #[inline]
    fn is_v_config_non_binary(v_config: &[usize]) -> bool {
        v_config.iter().any(|&digit| digit == 2)
    }

    #[inline]
    pub fn compute_and_update_tA_inplace_generic<const NUM_SVO_ROUNDS: usize, F: JoltField>(
        binary_az_evals: &[i128],
        binary_bz_evals: &[i128],
        e_in_val: &F,
        temp_tA: &mut [F],
    ) {
        match NUM_SVO_ROUNDS {
            1 => {
                compute_and_update_tA_inplace_1(binary_az_evals, binary_bz_evals, e_in_val, temp_tA)
            }
            2 => {
                compute_and_update_tA_inplace_2(binary_az_evals, binary_bz_evals, e_in_val, temp_tA)
            }
            3 => {
                compute_and_update_tA_inplace_3(binary_az_evals, binary_bz_evals, e_in_val, temp_tA)
            }
            _ => compute_and_update_tA_inplace::<NUM_SVO_ROUNDS, F>(binary_az_evals, binary_bz_evals, e_in_val, temp_tA),
        }
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
        debug_assert!(binary_az_evals.len() == 2);
        debug_assert!(binary_bz_evals.len() == 2);
        debug_assert!(temp_tA.len() == 1);
        let az_I = binary_az_evals[1] - binary_az_evals[0];
        if az_I != 0 {
            let bz_I = binary_bz_evals[1] - binary_bz_evals[0];
            if bz_I != 0 {
                let product_i128 = az_I
                    .checked_mul(bz_I)
                    .expect("Az_I*Bz_I product overflow i128");
                temp_tA[0] += e_in_val.mul_i128(product_i128);
            }
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
        debug_assert!(binary_az_evals.len() == 4);
        debug_assert!(binary_bz_evals.len() == 4);
        debug_assert!(temp_tA.len() == 5);
        // Binary evaluations: (Y0,Y1) -> index Y0*2 + Y1
        let az00 = binary_az_evals[0];
        let bz00 = binary_bz_evals[0]; // Y0=0, Y1=0
        let az01 = binary_az_evals[1];
        let bz01 = binary_bz_evals[1]; // Y0=0, Y1=1
        let az10 = binary_az_evals[2];
        let bz10 = binary_bz_evals[2]; // Y0=1, Y1=0
        let az11 = binary_az_evals[3];
        let bz11 = binary_bz_evals[3]; // Y0=1, Y1=1

        // Extended evaluations (points with at least one \'I\')
        // temp_tA indices follow the order: (0,I), (1,I), (I,0), (I,1), (I,I)

        // 1. Point (0,I) -> temp_tA[0]
        let az_0I = az01 - az00;
        let bz_0I = bz01 - bz00;
        if az_0I != 0 && bz_0I != 0 {
            let prod_0I = az_0I
                .checked_mul(bz_0I)
                .expect("Product overflow for (0,I)");
            temp_tA[0] += e_in_val.mul_i128(prod_0I);
        }

        // 2. Point (1,I) -> temp_tA[1]
        let az_1I = az11 - az10;
        let bz_1I = bz11 - bz10;
        if az_1I != 0 && bz_1I != 0 {
            let prod_1I = az_1I
                .checked_mul(bz_1I)
                .expect("Product overflow for (1,I)");
            temp_tA[1] += e_in_val.mul_i128(prod_1I);
        }

        // 3. Point (I,0) -> temp_tA[2]
        let az_I0 = az10 - az00;
        if az_I0 != 0 {
            let bz_I0 = bz10 - bz00;
            if bz_I0 != 0 {
                let prod_I0 = az_I0
                    .checked_mul(bz_I0)
                    .expect("Product overflow for (I,0)");
                temp_tA[2] += e_in_val.mul_i128(prod_I0);
            }
        }

        // 4. Point (I,1) -> temp_tA[3]
        let az_I1 = az11 - az01;
        if az_I1 != 0 {
            let bz_I1 = bz11 - bz01;
            if bz_I1 != 0 {
                let prod_I1 = az_I1
                    .checked_mul(bz_I1)
                    .expect("Product overflow for (I,1)");
                temp_tA[3] += e_in_val.mul_i128(prod_I1);
            }
        }

        // 5. Point (I,I) -> temp_tA[4]
        let az_II = az_1I - az_0I;
        if az_II != 0 {
            let bz_II = bz_1I - bz_0I;
            if bz_II != 0 {
                let prod_II = az_II
                    .checked_mul(bz_II)
                    .expect("Product overflow for (I,I)");
                // At this point, pretty unlikely for prod_II to be 1
                temp_tA[4] += e_in_val.mul_i128(prod_II);
            }
        }
    }

    /// Special case when `num_svo_rounds == 3`
    /// In this case, we know that there are 8 binary evals of Az and Bz,
    /// corresponding to (0,0,0) => 0, (0,0,1) => 1, (0,1,0) => 2, (0,1,1) => 3, (1,0,0) => 4,
    /// (1,0,1) => 5, (1,1,0) => 6, (1,1,1) => 7.
    ///
    /// There are 27 - 8 = 19 temp_tA accumulators, with the following logic:
    /// (listed in increasing order, when considering ∞ as 2 and going from MSB to LSB)
    ///
    /// temp_tA[0,0,∞] += e_in_val * (az[0,0,1] - az[0,0,0]) * (bz[0,0,1] - bz[0,0,0])
    /// temp_tA[0,1,∞] += e_in_val * (az[0,1,1] - az[0,1,0]) * (bz[0,1,1] - bz[0,1,0])
    /// temp_tA[0,∞,0] += e_in_val * (az[0,1,0] - az[0,0,0]) * (bz[0,1,0] - bz[0,0,0])
    /// temp_tA[0,∞,1] += e_in_val * (az[0,1,1] - az[0,0,1]) * (bz[0,1,1] - bz[0,0,1])
    /// temp_tA[0,∞,∞] += e_in_val * (az[0,1,∞] - az[0,0,∞]) * (bz[0,1,∞] - bz[0,0,∞])
    ///
    /// temp_tA[1,0,∞] += e_in_val * (az[1,0,1] - az[1,0,0]) * (bz[1,0,1] - bz[1,0,0])
    /// temp_tA[1,1,∞] += e_in_val * (az[1,1,1] - az[1,1,0]) * (bz[1,1,1] - bz[1,1,0])
    /// temp_tA[1,∞,0] += e_in_val * (az[1,1,0] - az[1,0,0]) * (bz[1,1,0] - bz[1,0,0])
    /// temp_tA[1,∞,1] += e_in_val * (az[1,1,0] - az[1,0,0]) * (bz[1,1,0] - bz[1,0,0])
    /// temp_tA[1,∞,∞] += e_in_val * (az[1,1,∞] - az[1,0,∞]) * (bz[1,1,∞] - bz[1,0,∞])
    ///
    /// temp_tA[∞,0,0] += e_in_val * (az[1,0,0] - az[0,0,0]) * (bz[1,0,0] - bz[0,0,0])
    /// temp_tA[∞,0,1] += e_in_val * (az[1,0,1] - az[0,0,1]) * (bz[1,0,1] - bz[0,0,1])
    /// temp_tA[∞,0,∞] += e_in_val * (az[1,0,∞] - az[0,0,∞]) * (bz[1,0,∞] - bz[0,0,∞])
    /// temp_tA[∞,1,0] += e_in_val * (az[1,1,0] - az[0,1,0]) * (bz[1,1,0] - bz[0,1,0])
    /// temp_tA[∞,1,1] += e_in_val * (az[1,1,1] - az[0,1,1]) * (bz[1,1,1] - bz[0,1,1])
    /// temp_tA[∞,1,∞] += e_in_val * (az[1,1,∞] - az[0,1,∞]) * (bz[1,1,∞] - bz[0,1,∞])
    /// temp_tA[∞,∞,0] += e_in_val * (az[1,∞,0] - az[0,∞,0]) * (bz[1,∞,0] - bz[0,∞,0])
    /// temp_tA[∞,∞,1] += e_in_val * (az[1,∞,1] - az[0,∞,1]) * (bz[1,∞,1] - bz[0,∞,1])
    /// temp_tA[∞,∞,∞] += e_in_val * (az[1,∞,∞] - az[0,∞,∞]) * (bz[1,∞,∞] - bz[0,∞,∞])
    ///
    #[inline]
    pub fn compute_and_update_tA_inplace_3<F: JoltField>(
        binary_az_evals: &[i128],
        binary_bz_evals: &[i128],
        e_in_val: &F,
        temp_tA: &mut [F],
    ) {
        debug_assert!(binary_az_evals.len() == 8);
        debug_assert!(binary_bz_evals.len() == 8);
        debug_assert!(temp_tA.len() == 19);

        // Binary evaluations (Y0,Y1,Y2) -> index Y0*4 + Y1*2 + Y2
        let az000 = binary_az_evals[0];
        let bz000 = binary_bz_evals[0];
        let az001 = binary_az_evals[1];
        let bz001 = binary_bz_evals[1];
        let az010 = binary_az_evals[2];
        let bz010 = binary_bz_evals[2];
        let az011 = binary_az_evals[3];
        let bz011 = binary_bz_evals[3];
        let az100 = binary_az_evals[4];
        let bz100 = binary_bz_evals[4];
        let az101 = binary_az_evals[5];
        let bz101 = binary_bz_evals[5];
        let az110 = binary_az_evals[6];
        let bz110 = binary_bz_evals[6];
        let az111 = binary_az_evals[7];
        let bz111 = binary_bz_evals[7];
        
        // Populate temp_tA in lexicographical MSB-first order
        // Z=0, O=1, I=2 for Y_i
        
        // Point (0,0,I) -> temp_tA[0]
        let az_00I = az001 - az000;
        let bz_00I = bz001 - bz000;
        if az_00I != 0 && bz_00I != 0 {
            let prod = az_00I.checked_mul(bz_00I).expect("Prod overflow");
            // Test `mul_i128_1_optimized`
            temp_tA[0] += e_in_val.mul_i128_1_optimized(prod);
        }

        // Point (0,1,I) -> temp_tA[1]
        let az_01I = az011 - az010;
        let bz_01I = bz011 - bz010;
        if az_01I != 0 && bz_01I != 0 {
            let prod = az_01I.checked_mul(bz_01I).expect("Prod overflow");
            temp_tA[1] += e_in_val.mul_i128(prod);
        }
        
        // Point (0,I,0) -> temp_tA[2]
        let az_0I0 = az010 - az000;
        let bz_0I0 = bz010 - bz000;
        if az_0I0 != 0 && bz_0I0 != 0 {
            let prod = az_0I0.checked_mul(bz_0I0).expect("Prod overflow");
            temp_tA[2] += e_in_val.mul_i128(prod);
        }
        
        // Point (0,I,1) -> temp_tA[3]
        let az_0I1 = az011 - az001;
        let bz_0I1 = bz011 - bz001;
        if az_0I1 != 0 && bz_0I1 != 0 {
            let prod = az_0I1.checked_mul(bz_0I1).expect("Prod overflow");
            temp_tA[3] += e_in_val.mul_i128(prod);
        }
        
        // Point (0,I,I) -> temp_tA[4]
        let az_0II = az_01I - az_00I;
        let bz_0II = bz_01I - bz_00I; // Need to compute this outside for III term
        if az_0II != 0 && bz_0II != 0 {
            let prod = az_0II.checked_mul(bz_0II).expect("Prod overflow");
            temp_tA[4] += e_in_val.mul_i128(prod);
        }
        
        // Point (1,0,I) -> temp_tA[5]
        let az_10I = az101 - az100;
        let bz_10I = bz101 - bz100;
        if az_10I != 0 && bz_10I != 0 {
            let prod = az_10I.checked_mul(bz_10I).expect("Prod overflow");
            temp_tA[5] += e_in_val.mul_i128(prod);
        }
        
        // Point (1,1,I) -> temp_tA[6]
        let az_11I = az111 - az110;
        let bz_11I = bz111 - bz110;
        if az_11I != 0 && bz_11I != 0 {
            let prod = az_11I.checked_mul(bz_11I).expect("Prod overflow");
            temp_tA[6] += e_in_val.mul_i128(prod);
        }

        // Point (1,I,0) -> temp_tA[7]
        let az_1I0 = az110 - az100;
        let bz_1I0 = bz110 - bz100;
        if az_1I0 != 0 && bz_1I0 != 0 {
            let prod = az_1I0.checked_mul(bz_1I0).expect("Prod overflow");
            temp_tA[7] += e_in_val.mul_i128(prod);
        }

        // Point (1,I,1) -> temp_tA[8]
        let az_1I1 = az111 - az101;
        let bz_1I1 = bz111 - bz101;
        if az_1I1 != 0 && bz_1I1 != 0 {
            let prod = az_1I1.checked_mul(bz_1I1).expect("Prod overflow");
            temp_tA[8] += e_in_val.mul_i128(prod);
        }

        // Point (1,I,I) -> temp_tA[9]
        let az_1II = az_11I - az_10I;
        let bz_1II = bz_11I - bz_10I; // Need to compute this outside for III term
        if az_1II != 0 && bz_1II != 0 {
            let prod = az_1II.checked_mul(bz_1II).expect("Prod overflow");
            temp_tA[9] += e_in_val.mul_i128(prod);
        }

        // Point (I,0,0) -> temp_tA[10]
        let az_I00 = az100 - az000;
        let bz_I00 = bz100 - bz000;
        if az_I00 != 0 && bz_I00 != 0 {
            let prod = az_I00.checked_mul(bz_I00).expect("Prod overflow");
            temp_tA[10] += e_in_val.mul_i128(prod);
        }

        // Point (I,0,1) -> temp_tA[11]
        let az_I01 = az101 - az001;
        let bz_I01 = bz101 - bz001;
        if az_I01 != 0 && bz_I01 != 0 {
                let prod = az_I01.checked_mul(bz_I01).expect("Prod overflow");
                temp_tA[11] += e_in_val.mul_i128(prod);
        }

        // Point (I,0,I) -> temp_tA[12]
        let az_I0I = az_I01 - az_I00; // Uses precomputed az_I01, az_I00
        if az_I0I != 0 {
            let bz_I0I = bz_I01 - bz_I00; // Uses precomputed bz_I01, bz_I00
            if bz_I0I != 0 {
                let prod = az_I0I.checked_mul(bz_I0I).expect("Prod overflow");
                temp_tA[12] += e_in_val.mul_i128(prod);
            }
        }

        // Point (I,1,0) -> temp_tA[13]
        let az_I10 = az110 - az010;
        let bz_I10 = bz110 - bz010;
        if az_I10 != 0 && bz_I10 != 0 {
            let prod = az_I10.checked_mul(bz_I10).expect("Prod overflow");
            temp_tA[13] += e_in_val.mul_i128(prod);
        }

        // Point (I,1,1) -> temp_tA[14]
        let az_I11 = az111 - az011;
        let bz_I11 = bz111 - bz011;
        if az_I11 != 0 && bz_I11 != 0 {
            let prod = az_I11.checked_mul(bz_I11).expect("Prod overflow");
            temp_tA[14] += e_in_val.mul_i128(prod);
        }

        // Point (I,1,I) -> temp_tA[15]
        let az_I1I = az_I11 - az_I10; // Uses precomputed az_I11, az_I10
        if az_I1I != 0 {
            let bz_I1I = bz_I11 - bz_I10; // Uses precomputed bz_I11, bz_I10
            if bz_I1I != 0 {
                let prod = az_I1I.checked_mul(bz_I1I).expect("Prod overflow");
                temp_tA[15] += e_in_val.mul_i128(prod);
            }
        }

        // Point (I,I,0) -> temp_tA[16]
        let az_II0 = az_1I0 - az_0I0; // Uses precomputed az_1I0, az_0I0
        if az_II0 != 0 {
            let bz_II0 = bz_1I0 - bz_0I0; // Uses precomputed bz_1I0, bz_0I0
            if bz_II0 != 0 {
                let prod = az_II0.checked_mul(bz_II0).expect("Prod overflow");
                temp_tA[16] += e_in_val.mul_i128(prod);
            }
        }

        // Point (I,I,1) -> temp_tA[17]
        let az_II1 = az_1I1 - az_0I1; // Uses precomputed az_1I1, az_0I1
        if az_II1 != 0 {
            let bz_II1 = bz_1I1 - bz_0I1; // Uses precomputed bz_1I1, bz_0I1
            if bz_II1 != 0 {
                let prod = az_II1.checked_mul(bz_II1).expect("Prod overflow");
                temp_tA[17] += e_in_val.mul_i128(prod);
            }
        }

        // Point (I,I,I) -> temp_tA[18]
        // az_III depends on az_1II and az_0II.
        // az_1II was computed for temp_tA[9]
        // az_0II was computed for temp_tA[4]
        let az_III = az_1II - az_0II;
        if az_III != 0 {
            // bz_III depends on bz_1II and bz_0II.
            // bz_1II was computed for temp_tA[9]
            // bz_0II was computed for temp_tA[4]
            let bz_III = bz_1II - bz_0II;
            if bz_III != 0 {
                let prod = az_III.checked_mul(bz_III).expect("Prod overflow");
                temp_tA[18] += e_in_val.mul_i128(prod);
            }
        }
    }

    /// Performs in-place multilinear extension on ternary evaluation vectors
    /// and updates the temporary accumulator `temp_tA` with the product contribution.
    /// This is the generic version with no bound on num_svo_rounds.
    /// TODO: optimize this further
    #[inline]
    pub fn compute_and_update_tA_inplace<const NUM_SVO_ROUNDS: usize, F: JoltField>(
        binary_az_evals_input: &[i128], // Source of 2^N binary evals for Az
        binary_bz_evals_input: &[i128], // Source of 2^N binary evals for Bz
        e_in_val: &F,
        temp_tA: &mut [F],                  // Target for 3^N - 2^N non-binary extended products
    ) {
        if NUM_SVO_ROUNDS == 0 {
            debug_assert!(temp_tA.is_empty(), "temp_tA should be empty for 0 SVO rounds");
            return;
        }

        let num_binary_points = 1 << NUM_SVO_ROUNDS;
        let num_ternary_points = 3_usize.pow(NUM_SVO_ROUNDS as u32);

        debug_assert_eq!(binary_az_evals_input.len(), num_binary_points, "binary_az_evals_input length mismatch");
        debug_assert_eq!(binary_bz_evals_input.len(), num_binary_points, "binary_bz_evals_input length mismatch");
        debug_assert_eq!(temp_tA.len(), num_ternary_points - num_binary_points, "temp_tA length mismatch");

        // These will store all 3^N extended evaluations.
        // Using Vec here as heap allocation is acceptable for this setup phase.
        // The core logic avoids allocations.
        let mut extended_az_evals: Vec<i128> = vec![0; num_ternary_points];
        let mut extended_bz_evals: Vec<i128> = vec![0; num_ternary_points];

        let mut current_temp_tA_idx = 0;

        // Iterate k from 0 to 3^NUM_SVO_ROUNDS - 1.
        // k represents a point in the ternary hypercube (y_0, ..., y_{N-1})
        // where y_i are digits of k in base 3 (0=Zero, 1=One, 2=Infinity).
        // The iteration order is lexicographical if we consider (y_0, ..., y_{N-1}) MSB-first.
        for k_ternary_idx in 0..num_ternary_points {
            let mut temp_k = k_ternary_idx;
            let mut is_binary_point_flag = true;
            let mut first_inf_dim_msb_idx: Option<usize> = None; // Dimension index (0 to N-1, MSB-first) of the most significant Infinity

            // Decompose k_ternary_idx into base-3 digits to represent the ternary point.
            // We store it in an array where index 0 is MSB.
            let mut current_coords_base3 = [0u8; NUM_SVO_ROUNDS];
            for i_dim_rev in 0..NUM_SVO_ROUNDS { // Iterates to fill from LSB of k
                let i_dim_msb = NUM_SVO_ROUNDS - 1 - i_dim_rev; // Current dimension index (MSB is 0)
                let digit = (temp_k % 3) as u8;
                current_coords_base3[i_dim_msb] = digit;
                if digit == 2 { // Digit 2 represents Infinity
                    is_binary_point_flag = false;
                    if first_inf_dim_msb_idx.is_none() {
                        first_inf_dim_msb_idx = Some(i_dim_msb);
                    }
                }
                temp_k /= 3;
            }

            // Calculate extended_az_evals[k_ternary_idx] and extended_bz_evals[k_ternary_idx]
            if is_binary_point_flag {
                // Convert current_coords_base3 (containing only 0s and 1s) to a binary index
                let mut binary_idx = 0;
                for i_dim_msb in 0..NUM_SVO_ROUNDS {
                    binary_idx <<= 1;
                    if current_coords_base3[i_dim_msb] == 1 {
                        binary_idx |= 1;
                    }
                }
                extended_az_evals[k_ternary_idx] = binary_az_evals_input[binary_idx];
                extended_bz_evals[k_ternary_idx] = binary_bz_evals_input[binary_idx];
            } else {
                // Non-binary point: calculate using P(...,Inf,...) = P(...,1,...) - P(...,0,...)
                // The terms on the RHS are at indices < k_ternary_idx.
                let j_inf_dim = first_inf_dim_msb_idx.unwrap(); // Safe due to !is_binary_point_flag

                // Power of 3 for the dimension j_inf_dim (0-indexed from MSB)
                // Example: N=3, j_inf_dim=0 (MSB), power = 3^(3-1-0) = 3^2 = 9
                //          N=3, j_inf_dim=1 (Mid), power = 3^(3-1-1) = 3^1 = 3
                //          N=3, j_inf_dim=2 (LSB), power = 3^(3-1-2) = 3^0 = 1
                let inf_dim_power_of_3 = 3_usize.pow((NUM_SVO_ROUNDS - 1 - j_inf_dim) as u32);

                // k_ternary_idx has a \'2\' at dimension j_inf_dim.
                // k_at_1_idx corresponds to changing this \'2\' to a \'1\'.
                // k_at_0_idx corresponds to changing this \'2\' to a \'0\'.
                let k_at_1_idx = k_ternary_idx - inf_dim_power_of_3;
                let k_at_0_idx = k_ternary_idx - 2 * inf_dim_power_of_3;
                
                debug_assert!(k_at_1_idx < k_ternary_idx, "k_at_1_idx should be less than k_ternary_idx");
                debug_assert!(k_at_0_idx < k_ternary_idx, "k_at_0_idx should be less than k_ternary_idx");

                extended_az_evals[k_ternary_idx] = extended_az_evals[k_at_1_idx] - extended_az_evals[k_at_0_idx];
                extended_bz_evals[k_ternary_idx] = extended_bz_evals[k_at_1_idx] - extended_bz_evals[k_at_0_idx];

                // This is a non-binary point, so it contributes to temp_tA.
                // The temp_tA array is indexed by the order non-binary points are encountered.
                debug_assert!(current_temp_tA_idx < temp_tA.len(), "temp_tA index out of bounds");

                let az_val = extended_az_evals[k_ternary_idx];
                if az_val != 0 {
                    let bz_val = extended_bz_evals[k_ternary_idx];
                    if bz_val != 0 {
                        let product = az_val.checked_mul(bz_val)
                            .expect("Extended Az*Bz product overflow i128");
                        temp_tA[current_temp_tA_idx] += e_in_val.mul_i128(product);
                    }
                }
                current_temp_tA_idx += 1;
            }
        }
        // Final check that all entries in temp_tA that were supposed to be filled, were.
        debug_assert_eq!(current_temp_tA_idx, temp_tA.len(), "temp_tA was not fully populated or was overpopulated.");
    }

    /// Generic version for distributing tA to svo accumulators
    #[inline]
    pub fn distribute_tA_to_svo_accumulators_generic<const NUM_SVO_ROUNDS: usize, F: JoltField>(
        tA_accums: &[F],
        x_out_val: usize,
        E_out_vec: &[Vec<F>],
        accums_zero: &mut [F],
        accums_infty: &mut [F],
    ) {
        match NUM_SVO_ROUNDS {
            1 => distribute_tA_to_svo_accumulators_1(
                tA_accums,
                x_out_val,
                E_out_vec,
                accums_zero,
                accums_infty,
            ),
            2 => distribute_tA_to_svo_accumulators_2(
                tA_accums,
                x_out_val,
                E_out_vec,
                accums_zero,
                accums_infty,
            ),
            3 => distribute_tA_to_svo_accumulators_3(
                tA_accums,
                x_out_val,
                E_out_vec,
                accums_zero,
                accums_infty,
            ),
            _ => panic!("Unsupported number of SVO rounds"),
        }
    }

    /// Hardcoded version for `num_svo_rounds == 1`
    /// We have only one non-binary point (Y0=I), mapping to accum_1(I)
    #[inline]
    pub fn distribute_tA_to_svo_accumulators_1<F: JoltField>(
        tA_accums: &[F],
        x_out_val: usize,
        E_out_vec: &[Vec<F>],
        _accums_zero: &mut [F],
        accums_infty: &mut [F],
    ) {
        debug_assert!(_accums_zero.len() == 0);
        debug_assert!(accums_infty.len() == 1);
        debug_assert!(tA_accums.len() == 1);

        accums_infty[0] += E_out_vec[0][x_out_val] * tA_accums[0];
    }

    /// Hardcoded version for `num_svo_rounds == 2`
    /// We have 5 non-binary points with their corresponding mappings: (recall, LSB is rightmost)
    /// tA_accums indices:
    ///   [0]: Y_ext = (0,I)
    ///   [1]: Y_ext = (1,I)
    ///   [2]: Y_ext = (I,0)
    ///   [3]: Y_ext = (I,1)
    ///   [4]: Y_ext = (I,I)
    ///
    /// Target flat accumulators (NUM_SVO_ROUNDS = 2):
    ///   accums_zero (len 1): [ A_1(0,I) ]
    ///   accums_infty (len 4): [ A_0(empty,I), A_1(I,0), A_1(I,1), A_1(I,I) ]
    /// Note that A_0(empty,I) should receive contributions from (0,I) and (1,I).
    #[inline]
    pub fn distribute_tA_to_svo_accumulators_2<F: JoltField>(
        tA_accums: &[F],
        x_out_val: usize,
        E_out_vec: &[Vec<F>],
        accums_zero: &mut [F],
        accums_infty: &mut [F],
    ) {
        debug_assert!(tA_accums.len() == 5);
        debug_assert!(accums_zero.len() == 1);
        debug_assert!(accums_infty.len() == 4);
        debug_assert!(E_out_vec.len() >= 2);

        let E0_y0 = E_out_vec[0][(x_out_val << 1) | 0];
        let E0_y1 = E_out_vec[0][(x_out_val << 1) | 1];
        let E1_yempty = E_out_vec[1][x_out_val];

        // Y_ext = (0,I) -> tA_accums[0]
        // Contributes to A_0(empty,I) (i.e. accums_infty[0]) via E_out_0[x_out_val | 0] and
        // A_1(0,I) (i.e. accums_zero[0]) via E_out_1[x_out_val]

        accums_infty[0] += E0_y0 * tA_accums[0];

        accums_zero[0] += E1_yempty * tA_accums[0];

        // Y_ext = (1,I) -> tA_accums[1]
        // Contributes to A_0(empty,I) (i.e. accums_infty[0]) via E_out_0[x_out_val | 1]
        // (no term A_1(1,I) as it is not needed i.e. it\'s an eval at 1)
        accums_infty[0] += E0_y1 * tA_accums[1];

        // Y_ext = (I,0) -> tA_accums[2]
        // Contributes to A_1(I,0) (i.e. accums_infty[1]) via E_out_1[x_out_val]
        accums_infty[1] += E1_yempty * tA_accums[2];

        // Y_ext = (I,1) -> tA_accums[3]
        // Contributes to A_1(I,1) (i.e. accums_infty[2]) via E_out_1[x_out_val]
        accums_infty[2] += E1_yempty * tA_accums[3];

        // Y_ext = (I,I) -> tA_accums[4]
        // Contributes to A_1(I,I) (i.e. accums_infty[3]) via E_out_1[x_out_val]
        accums_infty[3] += E1_yempty * tA_accums[4];
    }

    /// Hardcoded version for `num_svo_rounds == 3`
    ///
    #[inline]
    pub fn distribute_tA_to_svo_accumulators_3<F: JoltField>(
        tA_accums: &[F],
        x_out_val: usize,
        E_out_vec: &[Vec<F>],
        accums_zero: &mut [F],
        accums_infty: &mut [F],
    ) {
        debug_assert!(tA_accums.len() == 19);
        debug_assert!(accums_zero.len() == 6);
        debug_assert!(accums_infty.len() == 13);
        debug_assert!(E_out_vec.len() >= 3);

        use SVOEvalPoint::{Infinity, One, Zero};

        // Accumulator slots (conceptual LSB-first paper round s_p)
        // Nomenclature: A_{s_p}(v_{s_p-1},...,v_0, u_s_p)
        // accums_infty slots
        const ACCUM_IDX_A0_I: usize = 0; // A_0(I)
        const ACCUM_IDX_A1_0_I: usize = 1; // A_1(v0=0, I)
        const ACCUM_IDX_A1_1_I: usize = 2; // A_1(v0=1, I)
        const ACCUM_IDX_A1_I_I: usize = 3; // A_1(v0=I, I)
        const ACCUM_IDX_A2_00_I: usize = 4; // A_2(v1=0,v0=0,I)
        const ACCUM_IDX_A2_01_I: usize = 5; // A_2(v1=0,v0=1,I)
        const ACCUM_IDX_A2_0I_I: usize = 6; // A_2(v1=0,v0=I,I)
        const ACCUM_IDX_A2_10_I: usize = 7; // A_2(v1=1,v0=0,I)
        const ACCUM_IDX_A2_11_I: usize = 8; // A_2(v1=1,v0=1,I)
        const ACCUM_IDX_A2_1I_I: usize = 9; // A_2(v1=1,v0=I,I)
        const ACCUM_IDX_A2_I0_I: usize = 10; // A_2(v1=I,v0=0,I)
        const ACCUM_IDX_A2_I1_I: usize = 11; // A_2(v1=I,v0=1,I)
        const ACCUM_IDX_A2_II_I: usize = 12; // A_2(v1=I,v0=I,I)

        // accums_zero slots
        const ACCUM_IDX_A1_I_0: usize = 0; // A_1(v0=I, 0)
        const ACCUM_IDX_A2_0I_0: usize = 1; // A_2(v1=0,v0=I,0)
        const ACCUM_IDX_A2_1I_0: usize = 2; // A_2(v1=1,v0=I,0)
        const ACCUM_IDX_A2_I0_0: usize = 3; // A_2(v1=I,v0=0,0)
        const ACCUM_IDX_A2_I1_0: usize = 4; // A_2(v1=I,v0=1,0)
        const ACCUM_IDX_A2_II_0: usize = 5; // A_2(v1=I,v0=I,0)

        // E_out_vec[s_code] interpretation (s_code is MSB-first index for Y_c variables):
        // E_out_vec[0] (s_code=0 in E_out_vec): Y2_c is u_eff for E. y_suffix_eff=(Y0_c, Y1_c). Index (x_out << 2) | (Y0_c_bit << 1) | Y1_c_bit
        let e0_suf00 = E_out_vec[0][(x_out_val << 2) | 0b00]; // y_suffix_eff=(Y0_c=0, Y1_c=0)
        let e0_suf01 = E_out_vec[0][(x_out_val << 2) | 0b01]; // y_suffix_eff=(Y0_c=0, Y1_c=1)
        let e0_suf10 = E_out_vec[0][(x_out_val << 2) | 0b10]; // y_suffix_eff=(Y0_c=1, Y1_c=0)
        let e0_suf11 = E_out_vec[0][(x_out_val << 2) | 0b11]; // y_suffix_eff=(Y0_c=1, Y1_c=1)

        // E_out_vec[1] (s_code=1 in E_out_vec): Y1_c is u_eff for E. y_suffix_eff=(Y0_c). Index (x_out << 1) | Y0_c_bit
        let e1_suf0 = E_out_vec[1][(x_out_val << 1) | 0]; // y_suffix_eff=(Y0_c=0)
        let e1_suf1 = E_out_vec[1][(x_out_val << 1) | 1]; // y_suffix_eff=(Y0_c=1)

        // E_out_vec[2] (s_code=2 in E_out_vec): Y0_c is u_eff for E. y_suffix_eff=(). Index x_out_val
        let e2_sufempty = E_out_vec[2][x_out_val];

        // Y_EXT_CODE_MAP[tA_idx] gives (Y0_c, Y1_c, Y2_c) for tA_accums[tA_idx]
        // Y0_c is MSB, Y2_c is LSB for code variables. This order matches compute_and_update_tA_inplace_3.
        const Y_EXT_CODE_MAP: [(SVOEvalPoint, SVOEvalPoint, SVOEvalPoint); 19] = [
            (Zero, Zero, Infinity),
            (Zero, One, Infinity),
            (Zero, Infinity, Zero),
            (Zero, Infinity, One),
            (Zero, Infinity, Infinity),
            (One, Zero, Infinity),
            (One, One, Infinity),
            (One, Infinity, Zero),
            (One, Infinity, One),
            (One, Infinity, Infinity),
            (Infinity, Zero, Zero),
            (Infinity, Zero, One),
            (Infinity, Zero, Infinity),
            (Infinity, One, Zero),
            (Infinity, One, One),
            (Infinity, One, Infinity),
            (Infinity, Infinity, Zero),
            (Infinity, Infinity, One),
            (Infinity, Infinity, Infinity),
        ];

        for i in 0..19 {
            let current_tA = tA_accums[i];
            // No !current_tA.is_zero() check, as requested

            let (y0_c, y1_c, y2_c) = Y_EXT_CODE_MAP[i]; // (MSB, Mid, LSB) from code\'s perspective

            // --- Contributions to Paper Round s_p=0 Accumulators (u_paper = Y2_c) ---
            // E-factor uses E_out_vec[0] (where Y2_c is u_eff), E_suffix is (Y0_c, Y1_c)
            if y2_c == Infinity {
                // u_paper = I
                if y0_c != Infinity && y1_c != Infinity {
                    // Suffix (Y0_c,Y1_c) for E_out_vec[0] must be binary
                    let e_val = match (y0_c, y1_c) {
                        (Zero, Zero) => e0_suf00,
                        (Zero, One) => e0_suf01,
                        (One, Zero) => e0_suf10,
                        (One, One) => e0_suf11,
                        _ => unreachable!(), // Should be covered by binary check above
                    };
                    accums_infty[ACCUM_IDX_A0_I] += current_tA * e_val;
                }
            }
            // No A_0(0) slots defined in the provided consts for accums_zero.

            // --- Contributions to Paper Round s_p=1 Accumulators (u_paper = Y1_c, v0_paper = Y2_c) ---
            // E-factor uses E_out_vec[1] (where Y1_c is u_eff), E_suffix is (Y0_c)
            if y0_c != Infinity {
                // Suffix Y0_c for E_out_vec[1] must be binary
                let e1_val = if y0_c == Zero { e1_suf0 } else { e1_suf1 }; // y0_c is One or Zero here

                if y1_c == Infinity {
                    // u_paper = I
                    match y2_c {
                        // v0_paper = Y2_c
                        Zero => {
                            accums_infty[ACCUM_IDX_A1_0_I] += current_tA * e1_val;
                        }
                        One => {
                            accums_infty[ACCUM_IDX_A1_1_I] += current_tA * e1_val;
                        }
                        Infinity => {
                            accums_infty[ACCUM_IDX_A1_I_I] += current_tA * e1_val;
                        }
                    }
                } else if y1_c == Zero {
                    // u_paper = 0
                    if y2_c == Infinity {
                        // v0_paper = I, for A_1(I,0)
                        accums_zero[ACCUM_IDX_A1_I_0] += current_tA * e1_val;
                    }
                }
            }

            // --- Contributions to Paper Round s_p=2 Accumulators (u_paper = Y0_c, v_paper = (Y1_c,Y2_c)) ---
            // E-factor uses E_out_vec[2] (where Y0_c is u_eff), E_suffix is empty
            let e2_val = e2_sufempty;
            if y0_c == Infinity {
                // u_paper = I
                match (y1_c, y2_c) {
                    // v_paper = (Y1_c, Y2_c)
                    (Zero, Zero) => {
                        accums_infty[ACCUM_IDX_A2_00_I] += current_tA * e2_val;
                    }
                    (Zero, One) => {
                        accums_infty[ACCUM_IDX_A2_01_I] += current_tA * e2_val;
                    }
                    (Zero, Infinity) => {
                        accums_infty[ACCUM_IDX_A2_0I_I] += current_tA * e2_val;
                    }
                    (One, Zero) => {
                        accums_infty[ACCUM_IDX_A2_10_I] += current_tA * e2_val;
                    }
                    (One, One) => {
                        accums_infty[ACCUM_IDX_A2_11_I] += current_tA * e2_val;
                    }
                    (One, Infinity) => {
                        accums_infty[ACCUM_IDX_A2_1I_I] += current_tA * e2_val;
                    }
                    (Infinity, Zero) => {
                        accums_infty[ACCUM_IDX_A2_I0_I] += current_tA * e2_val;
                    }
                    (Infinity, One) => {
                        accums_infty[ACCUM_IDX_A2_I1_I] += current_tA * e2_val;
                    }
                    (Infinity, Infinity) => {
                        accums_infty[ACCUM_IDX_A2_II_I] += current_tA * e2_val;
                    }
                }
            } else if y0_c == Zero {
                // u_paper = 0
                match (y1_c, y2_c) {
                    // v_paper = (Y1_c, Y2_c)
                    // Only specific v_paper configs for A_2(v,0) are stored, based on ACCUM_IDX constants
                    (Zero, Infinity) => {
                        accums_zero[ACCUM_IDX_A2_0I_0] += current_tA * e2_val;
                    }
                    (One, Infinity) => {
                        accums_zero[ACCUM_IDX_A2_1I_0] += current_tA * e2_val;
                    }
                    (Infinity, Zero) => {
                        accums_zero[ACCUM_IDX_A2_I0_0] += current_tA * e2_val;
                    }
                    (Infinity, One) => {
                        accums_zero[ACCUM_IDX_A2_I1_0] += current_tA * e2_val;
                    }
                    (Infinity, Infinity) => {
                        accums_zero[ACCUM_IDX_A2_II_0] += current_tA * e2_val;
                    }
                    _ => {} // Other v_paper configs (e.g. fully binary, or other Infinity patterns) for u_paper=0 are not stored.
                }
            }
        }
    }

    // Distributes the accumulated tA values (sum over x_in) for a single x_out_val
    // to the appropriate SVO round accumulators.
    #[inline]
    pub fn distribute_tA_to_svo_accumulators<const NUM_SVO_ROUNDS: usize, F: JoltField>(
        tA_accums: &[F],
        x_out_val: usize,
        E_out_vec: &[Vec<F>],
        accums_zero: &mut [F],
        accums_infty: &mut [F],
    ) {
        if NUM_SVO_ROUNDS == 0 {
            debug_assert!(tA_accums.is_empty());
            debug_assert!(accums_zero.is_empty());
            debug_assert!(accums_infty.is_empty());
            return;
        }

        // --- Helper: Convert k (overall ternary index) to Y_ext coordinates (MSB-first SVOEvalPoint) ---
        fn k_to_y_ext<const N: usize>(k_ternary: usize) -> ([SVOEvalPoint; N], bool /*is_binary*/) {
            let mut coords = [SVOEvalPoint::Zero; N];
            let mut temp_k = k_ternary;
            let mut is_binary = true;
            if N > 0 { // Guard against N=0 for current_coords_base3 access
                for i_rev in 0..N {
                    let i_msb = N - 1 - i_rev; // Fill from MSB
                    coords[i_msb] = match temp_k % 3 {
                        0 => SVOEvalPoint::Zero,
                        1 => SVOEvalPoint::One,
                        2 => { is_binary = false; SVOEvalPoint::Infinity },
                        _ => unreachable!(),
                    };
                    temp_k /= 3;
                }
            }
            (coords, is_binary)
        }
        
        // --- Helper: Convert LSB-first SVOEvalPoint tuple to base-3 index ---
        // e.g., (LSB, ..., MSB) -> LSB*3^0 + ... + MSB*3^{len-1}
        fn v_tuple_to_base3_idx(v_tuple_lsb_ordered: &[SVOEvalPoint]) -> usize {
            let mut idx = 0;
            let mut power_of_3 = 1;
            for point in v_tuple_lsb_ordered {
                idx += match point {
                    SVOEvalPoint::Zero => 0,
                    SVOEvalPoint::One => power_of_3,
                    SVOEvalPoint::Infinity => 2 * power_of_3,
                };
                power_of_3 *= 3;
            }
            idx
        }

        // --- Helper: Check if LSB-first SVOEvalPoint tuple contains Infinity ---
        fn v_tuple_has_infinity(v_tuple_lsb_ordered: &[SVOEvalPoint]) -> bool {
            v_tuple_lsb_ordered.iter().any(|&p| p == SVOEvalPoint::Infinity)
        }
        
        // --- Helper: Convert LSB-first SVOEvalPoint tuple to non-binary base-3 index ---
        // This counts how many valid `v` configurations (non-binary, ternary) appear before this one.
        fn v_tuple_to_non_binary_base3_idx(v_tuple_lsb_ordered: &[SVOEvalPoint]) -> usize {
            let num_v_vars = v_tuple_lsb_ordered.len();
            if num_v_vars == 0 { 
                // This case means s_p = 0. The A_0(.;0) accumulator block has size 3^0 - 2^0 = 0.
                // So, an index is not meaningful here as there are no slots.
                // The calling code should check if num_slots_in_block_A_sp_zero > 0.
                return 0; 
            }

            let mut index_count = 0;
            let target_v_config_ternary_value = v_tuple_to_base3_idx(v_tuple_lsb_ordered);

            // Iterate all ternary configurations for v_tuple that are lexicographically smaller
            let max_k_v = 3_usize.pow(num_v_vars as u32);
            for k_v_ternary in 0..max_k_v {
                if k_v_ternary == target_v_config_ternary_value {
                    break; // Stop when we reach the target configuration itself
                }

                // Check if this k_v_ternary configuration for v has an infinity
                let mut current_k_v_has_inf = false;
                let mut temp_k = k_v_ternary;
                for _ in 0..num_v_vars {
                    if temp_k % 3 == 2 { // 2 represents Infinity
                        current_k_v_has_inf = true;
                        break;
                    }
                    temp_k /= 3;
                }

                if current_k_v_has_inf {
                    index_count += 1;
                }
            }
            index_count
        }


        // 1. Precompute E-factors: E_factors[s_e_idx][bin_suffix_idx]
        let mut precomputed_E_factors: Vec<Vec<F>> = Vec::with_capacity(NUM_SVO_ROUNDS);
        for s_e_idx in 0..NUM_SVO_ROUNDS { // s_e_idx is MSB-first index for Y_c, and for E_out_vec
            let num_suffix_vars_for_E = if NUM_SVO_ROUNDS > s_e_idx + 1 { NUM_SVO_ROUNDS - (s_e_idx + 1) } else { 0 };
            let num_binary_suffixes = 1 << num_suffix_vars_for_E;
            let mut current_s_e_factors = Vec::with_capacity(num_binary_suffixes);
            if E_out_vec.len() > s_e_idx && !E_out_vec[s_e_idx].is_empty() {
                for bin_suffix_val in 0..num_binary_suffixes {
                    let e_idx = (x_out_val << num_suffix_vars_for_E) | bin_suffix_val;
                    if e_idx < E_out_vec[s_e_idx].len() {
                         current_s_e_factors.push(E_out_vec[s_e_idx][e_idx]);
                    } else {
                        current_s_e_factors.push(F::zero()); 
                    }
                }
            } else {
                 for _ in 0..num_binary_suffixes { current_s_e_factors.push(F::zero());}
            }
            precomputed_E_factors.push(current_s_e_factors);
        }

        // 2. Precompute Accumulator Offsets
        let mut round_offsets_infty = vec![0; NUM_SVO_ROUNDS];
        let mut round_offsets_zero = vec![0; NUM_SVO_ROUNDS];
        if NUM_SVO_ROUNDS > 0 {
            let mut current_sum_infty = 0;
            let mut current_sum_zero = 0;
            for s_p in 0..NUM_SVO_ROUNDS { // s_p is paper round index (number of v\'s)
                round_offsets_infty[s_p] = current_sum_infty;
                round_offsets_zero[s_p] = current_sum_zero;
                current_sum_infty += 3_usize.pow(s_p as u32);
                current_sum_zero += 3_usize.pow(s_p as u32) - 2_usize.pow(s_p as u32);
            }
        }
        
        // Main Loop
        let mut current_tA_idx = 0;
        let num_total_ternary_points = 3_usize.pow(NUM_SVO_ROUNDS as u32);

        for k_overall_idx in 0..num_total_ternary_points {
            let (y_ext_msb, is_y_ext_binary) = k_to_y_ext::<NUM_SVO_ROUNDS>(k_overall_idx);

            if is_y_ext_binary {
                continue;
            }
            if current_tA_idx >= tA_accums.len() { 
                break; 
            }
            let current_tA_val = tA_accums[current_tA_idx];
            if current_tA_val.is_zero() { // Optimization: if tA is zero, it contributes nothing
                current_tA_idx += 1;
                continue;
            }

            // Inner Loop: Target Paper Round s_p (also used as MSB index for E_out_vec choice)
            for s_p_msb_idx_for_E in 0..NUM_SVO_ROUNDS { 
                // s_p_msb_idx_for_E is the MSB index of the Y_c variable that is u_eff for E_out_vec[s_p_msb_idx_for_E]
                // This also aligns with the paper\'s s_p if we map s_p=0 (A0) to E_out_vec[0], s_p=1 (A1) to E_out_vec[1] etc.
                
                // u_eff_for_this_E is y_ext_msb[s_p_msb_idx_for_E]
                // Suffix for E_out_vec[s_p_msb_idx_for_E] is (y_ext_msb[s_p_msb_idx_for_E+1], ..., y_ext_msb[N-1])
                let num_suffix_vars_for_this_E = if NUM_SVO_ROUNDS > s_p_msb_idx_for_E + 1 { NUM_SVO_ROUNDS - (s_p_msb_idx_for_E + 1) } else { 0 };
                let mut e_suffix_bin_idx = 0;
                let mut e_suffix_for_this_E_is_binary = true;
                for i_suffix_bit_msb in 0..num_suffix_vars_for_this_E {
                    let y_coord_for_suffix = y_ext_msb[s_p_msb_idx_for_E + 1 + i_suffix_bit_msb];
                    e_suffix_bin_idx <<= 1;
                    match y_coord_for_suffix {
                        SVOEvalPoint::Zero => { /* e_suffix_bin_idx |= 0; */ }
                        SVOEvalPoint::One => { e_suffix_bin_idx |= 1; }
                        SVOEvalPoint::Infinity => { e_suffix_for_this_E_is_binary = false; break; }
                    }
                }

                let e_factor = if e_suffix_for_this_E_is_binary && 
                                  precomputed_E_factors.len() > s_p_msb_idx_for_E && 
                                  precomputed_E_factors[s_p_msb_idx_for_E].len() > e_suffix_bin_idx {
                    precomputed_E_factors[s_p_msb_idx_for_E][e_suffix_bin_idx]
                } else {
                    F::zero()
                };

                if e_factor.is_zero() { // If E-factor is zero, no contribution
                    continue;
                }

                // Determine u_A and v_A for A_{s_p_msb_idx_for_E}
                // u_A is the (s_p_msb_idx_for_E)-th LSB variable of Y_ext.
                // The paper round s_p (number of v_vars) is s_p_msb_idx_for_E.
                let paper_s_p = s_p_msb_idx_for_E; 

                let u_A = y_ext_msb[NUM_SVO_ROUNDS - 1 - paper_s_p]; 
                
                let mut v_A_tuple_lsb_ordered: Vec<SVOEvalPoint> = Vec::with_capacity(paper_s_p);
                for i_v_lsb in 0..paper_s_p { // iterates from LSB of v_A up to MSB of v_A
                    v_A_tuple_lsb_ordered.push(y_ext_msb[NUM_SVO_ROUNDS - 1 - i_v_lsb]);
                }

                match u_A {
                    SVOEvalPoint::Infinity => {
                        let base_offset = round_offsets_infty[paper_s_p];
                        let idx_within_block = v_tuple_to_base3_idx(&v_A_tuple_lsb_ordered);
                        if base_offset + idx_within_block < accums_infty.len() {
                             accums_infty[base_offset + idx_within_block] += current_tA_val * e_factor;
                        }
                    }
                    SVOEvalPoint::Zero => {
                        if paper_s_p == 0 { 
                            continue; 
                        }
                        if v_tuple_has_infinity(&v_A_tuple_lsb_ordered) { 
                           let base_offset = round_offsets_zero[paper_s_p];
                           let num_slots_in_block_A_sp_zero = 3_usize.pow(paper_s_p as u32) - 2_usize.pow(paper_s_p as u32);
                           
                           if num_slots_in_block_A_sp_zero > 0 { 
                               let idx_within_block = v_tuple_to_non_binary_base3_idx(&v_A_tuple_lsb_ordered);
                               if base_offset + idx_within_block < accums_zero.len() && idx_within_block < num_slots_in_block_A_sp_zero {
                                    accums_zero[base_offset + idx_within_block] += current_tA_val * e_factor;
                               }
                           }
                        }
                    }
                    SVOEvalPoint::One => { /* Does not contribute */ }
                }
            }
            current_tA_idx += 1;
        }
        debug_assert_eq!(current_tA_idx, tA_accums.len(), "tA_accums not fully processed or over-processed.");
    }

    /// Process the first few sum-check rounds using small value optimization (SVO)
    /// We take in the pre-computed accumulator values, and use them to compute the quadratic
    /// evaluations (and thus cubic polynomials) for the first few sum-check rounds.
    pub fn process_svo_sumcheck_rounds<
        const NUM_SVO_ROUNDS: usize,
        F: JoltField,
        ProofTranscript: Transcript,
    >(
        accums_zero: &[F],
        accums_infty: &[F],
        r_challenges: &mut Vec<F>,
        round_polys: &mut Vec<CompressedUniPoly<F>>,
        claim: &mut F,
        transcript: &mut ProofTranscript,
        eq_poly: &mut GruenSplitEqPolynomial<F>,
    ) {
        // Assert lengths of accumulator slices based on NUM_SVO_ROUNDS
        let expected_accums_zero_len = num_accums_eval_zero(NUM_SVO_ROUNDS);
        let expected_accums_infty_len = num_accums_eval_infty(NUM_SVO_ROUNDS);
        assert_eq!(
            accums_zero.len(),
            expected_accums_zero_len,
            "accums_zero length mismatch"
        );
        assert_eq!(
            accums_infty.len(),
            expected_accums_infty_len,
            "accums_infty length mismatch"
        );

        let mut lagrange_coeffs: Vec<F> = vec![F::one()];
        let mut current_acc_zero_offset = 0;
        let mut current_acc_infty_offset = 0;

        for i in 0..NUM_SVO_ROUNDS {
            let mut quadratic_eval_0 = F::zero();
            let mut quadratic_eval_infty = F::zero();

            if USES_SMALL_VALUE_OPTIMIZATION {
                let num_vars_in_v_config = i; // v_config is (v_0, ..., v_{i-1})
                let num_lagrange_coeffs_for_round = three_pow(num_vars_in_v_config);

                // Compute quadratic_eval_infty
                let num_accs_infty_curr_round = three_pow(num_vars_in_v_config);
                if num_accs_infty_curr_round > 0 && current_acc_infty_offset + num_accs_infty_curr_round <= accums_infty.len() {
                    let accums_infty_slice = &accums_infty
                        [current_acc_infty_offset..current_acc_infty_offset + num_accs_infty_curr_round];
                    for k in 0..num_lagrange_coeffs_for_round {
                        if k < accums_infty_slice.len() && k < lagrange_coeffs.len() {
                            quadratic_eval_infty += accums_infty_slice[k] * lagrange_coeffs[k];
                        }
                    }
                }
                current_acc_infty_offset += num_accs_infty_curr_round;

                // Compute quadratic_eval_0
                let num_accs_zero_curr_round = if num_vars_in_v_config == 0 {
                    0 // 3^0 - 2^0 = 0
                } else {
                    three_pow(num_vars_in_v_config) - two_pow(num_vars_in_v_config)
                };

                if num_accs_zero_curr_round > 0 && current_acc_zero_offset + num_accs_zero_curr_round <= accums_zero.len() {
                    let accums_zero_slice = &accums_zero
                        [current_acc_zero_offset..current_acc_zero_offset + num_accs_zero_curr_round];
                    let mut non_binary_v_config_counter = 0;
                    for k_global in 0..num_lagrange_coeffs_for_round {
                        let v_config = get_v_config_digits(k_global, num_vars_in_v_config);
                        if is_v_config_non_binary(&v_config) {
                            if non_binary_v_config_counter < accums_zero_slice.len() && k_global < lagrange_coeffs.len() {
                                quadratic_eval_0 += accums_zero_slice[non_binary_v_config_counter]
                                    * lagrange_coeffs[k_global];
                                non_binary_v_config_counter += 1;
                            }
                        }
                    }
                }
                current_acc_zero_offset += num_accs_zero_curr_round;
            }

            let r_i = process_eq_sumcheck_round(
                (quadratic_eval_0, quadratic_eval_infty),
                eq_poly,
                round_polys,
                r_challenges,
                claim,
                transcript,
            );

            let lagrange_coeffs_r_i = [F::one() - r_i, r_i, r_i * (r_i - F::one())];

            if i < NUM_SVO_ROUNDS.saturating_sub(1) {
                lagrange_coeffs = lagrange_coeffs_r_i
                    .iter()
                    .flat_map(|lagrange_coeff| {
                        lagrange_coeffs
                            .iter()
                            .map(move |coeff| *lagrange_coeff * *coeff)
                    })
                    .collect();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::svo_helpers::*;
    use ark_bn254::Fr as TestField;
    use ark_ff::Zero;
    // use SVOEvalPoint::*; // Not strictly needed for these tests as SVOEvalPoint is internal.

    // Helper to calculate the number of entries in temp_tA: 3^N - 2^N
    const fn num_temp_tA_entries(num_svo_rounds: usize) -> usize {
        if num_svo_rounds == 0 {
            return 0;
        }
        let pow3 = 3_usize.pow(num_svo_rounds as u32);
        let pow2 = 2_usize.pow(num_svo_rounds as u32);
        pow3 - pow2
    }

    // Helper to calculate number of entries for accums_zero: sum_{i=0}^{N-1} (3^i - 2^i)
    const fn num_accums_zero_entries_test_helper(num_svo_rounds: usize) -> usize {
        let mut sum = 0;
        let mut i = 0;
        while i < num_svo_rounds {
            sum += 3_usize.pow(i as u32) - 2_usize.pow(i as u32);
            i += 1;
        }
        sum
    }

    // Helper to calculate number of entries for accums_infty: sum_{i=0}^{N-1} 3^i
    const fn num_accums_infty_entries_test_helper(num_svo_rounds: usize) -> usize {
        let mut sum = 0;
        let mut i = 0;
        while i < num_svo_rounds {
            sum += 3_usize.pow(i as u32);
            i += 1;
        }
        sum
    }

    fn test_consistency_for_num_rounds(num_svo_rounds: usize) {
        let num_binary_points = 1 << num_svo_rounds;
        let num_temp_tA = num_temp_tA_entries(num_svo_rounds);

        // Test with non-zero patterned data
        let binary_az_evals: Vec<i128> = (0..num_binary_points).map(|i| (i + 1) as i128 * 2).collect();
        let binary_bz_evals: Vec<i128> = (0..num_binary_points).map(|i| (i + 2) as i128 * 3).collect();
        let e_in_val = TestField::from(10u64);

        let mut temp_tA_new = vec![TestField::zero(); num_temp_tA];
        let mut temp_tA_old = vec![TestField::zero(); num_temp_tA];
        
        // Call the new general function (the one being tested)
        match num_svo_rounds {
            1 => compute_and_update_tA_inplace::<1, TestField>(&binary_az_evals, &binary_bz_evals, &e_in_val, &mut temp_tA_new),
            2 => compute_and_update_tA_inplace::<2, TestField>(&binary_az_evals, &binary_bz_evals, &e_in_val, &mut temp_tA_new),
            3 => compute_and_update_tA_inplace::<3, TestField>(&binary_az_evals, &binary_bz_evals, &e_in_val, &mut temp_tA_new),
            _ => panic!("Unsupported NUM_SVO_ROUNDS for this consistency test structure"),
        }

        // Call the old hardcoded function
        match num_svo_rounds {
            1 => compute_and_update_tA_inplace_1(&binary_az_evals, &binary_bz_evals, &e_in_val, &mut temp_tA_old),
            2 => compute_and_update_tA_inplace_2(&binary_az_evals, &binary_bz_evals, &e_in_val, &mut temp_tA_old),
            3 => compute_and_update_tA_inplace_3(&binary_az_evals, &binary_bz_evals, &e_in_val, &mut temp_tA_old),
            _ => panic!("Unsupported NUM_SVO_ROUNDS for this consistency test structure"),
        }
        assert_eq!(temp_tA_new, temp_tA_old, "Mismatch for NUM_SVO_ROUNDS = {} with patterned data", num_svo_rounds);

        // Test with all zeros for binary evals
        let binary_az_evals_zeros: Vec<i128> = vec![0; num_binary_points];
        let binary_bz_evals_zeros: Vec<i128> = vec![0; num_binary_points];
        let mut temp_tA_new_zeros = vec![TestField::zero(); num_temp_tA];
        let mut temp_tA_old_zeros = vec![TestField::zero(); num_temp_tA];

        match num_svo_rounds {
            1 => compute_and_update_tA_inplace::<1, TestField>(&binary_az_evals_zeros, &binary_bz_evals_zeros, &e_in_val, &mut temp_tA_new_zeros),
            2 => compute_and_update_tA_inplace::<2, TestField>(&binary_az_evals_zeros, &binary_bz_evals_zeros, &e_in_val, &mut temp_tA_new_zeros),
            3 => compute_and_update_tA_inplace::<3, TestField>(&binary_az_evals_zeros, &binary_bz_evals_zeros, &e_in_val, &mut temp_tA_new_zeros),
            _ => panic!("Unsupported NUM_SVO_ROUNDS for this consistency test structure"),
        }
        
        match num_svo_rounds {
            1 => compute_and_update_tA_inplace_1(&binary_az_evals_zeros, &binary_bz_evals_zeros, &e_in_val, &mut temp_tA_old_zeros),
            2 => compute_and_update_tA_inplace_2(&binary_az_evals_zeros, &binary_bz_evals_zeros, &e_in_val, &mut temp_tA_old_zeros),
            3 => compute_and_update_tA_inplace_3(&binary_az_evals_zeros, &binary_bz_evals_zeros, &e_in_val, &mut temp_tA_old_zeros),
            _ => panic!("Unsupported NUM_SVO_ROUNDS for this consistency test structure"),
        }
        assert_eq!(temp_tA_new_zeros, temp_tA_old_zeros, "Mismatch for NUM_SVO_ROUNDS = {} with zero data", num_svo_rounds);

        // Test with e_in_val = 0
        let e_in_val_zero = TestField::zero();
        let mut temp_tA_new_zero_e = vec![TestField::zero(); num_temp_tA];
        let mut temp_tA_old_zero_e = vec![TestField::zero(); num_temp_tA];

        match num_svo_rounds {
            1 => compute_and_update_tA_inplace::<1, TestField>(&binary_az_evals, &binary_bz_evals, &e_in_val_zero, &mut temp_tA_new_zero_e),
            2 => compute_and_update_tA_inplace::<2, TestField>(&binary_az_evals, &binary_bz_evals, &e_in_val_zero, &mut temp_tA_new_zero_e),
            3 => compute_and_update_tA_inplace::<3, TestField>(&binary_az_evals, &binary_bz_evals, &e_in_val_zero, &mut temp_tA_new_zero_e),
            _ => panic!("Unsupported NUM_SVO_ROUNDS for this consistency test structure"),
        }

        match num_svo_rounds {
            1 => compute_and_update_tA_inplace_1(&binary_az_evals, &binary_bz_evals, &e_in_val_zero, &mut temp_tA_old_zero_e),
            2 => compute_and_update_tA_inplace_2(&binary_az_evals, &binary_bz_evals, &e_in_val_zero, &mut temp_tA_old_zero_e),
            3 => compute_and_update_tA_inplace_3(&binary_az_evals, &binary_bz_evals, &e_in_val_zero, &mut temp_tA_old_zero_e),
            _ => panic!("Unsupported NUM_SVO_ROUNDS for this consistency test structure"),
        }
         assert_eq!(temp_tA_new_zero_e, temp_tA_old_zero_e, "Mismatch for NUM_SVO_ROUNDS = {} with zero e_in_val", num_svo_rounds);
         for val in temp_tA_new_zero_e { // Confirm they are all zero
            assert!(val.is_zero(), "temp_tA should be all zeros if e_in_val is zero");
        }
    }

    #[test]
    fn test_compute_and_update_tA_inplace_vs_hardcoded_1() {
        test_consistency_for_num_rounds(1);
    }

    #[test]
    fn test_compute_and_update_tA_inplace_vs_hardcoded_2() {
        test_consistency_for_num_rounds(2);
    }

    #[test]
    fn test_compute_and_update_tA_inplace_vs_hardcoded_3() {
        test_consistency_for_num_rounds(3);
    }

    // --- Tests for distribute_tA_to_svo_accumulators --- 

    fn test_distribute_consistency(num_svo_rounds: usize) {
        let num_tA = num_temp_tA_entries(num_svo_rounds);
        let tA_accums: Vec<TestField> = (0..num_tA).map(|i| TestField::from((i + 1) as u64 * 100)).collect();
        
        let x_out_val = 1; // Arbitrary x_out_val for testing E_out_vec indexing

        // Mock E_out_vec: E_out_vec[s_code_E][(x_out_val << num_suffix) | suffix_val]
        // For simplicity in test, make E_out_vec values somewhat predictable.
        // E_out_vec[s_code_E] will have 2^(N-1-s_code_E) * (x_out_max_val_mock+1) entries.
        // Let x_out_val be small, and E_out_vec constructed to be dense enough.
        let mut E_out_vec: Vec<Vec<TestField>> = Vec::with_capacity(num_svo_rounds);
        for s_e in 0..num_svo_rounds {
            let num_suffix_vars = if num_svo_rounds > s_e + 1 { num_svo_rounds - (s_e + 1) } else { 0 };
            let num_e_entries_for_x_out = 1 << num_suffix_vars;
            let total_e_entries = (x_out_val + 10) * num_e_entries_for_x_out; // Ensure enough space for x_out_val
            let mut e_s: Vec<TestField> = Vec::with_capacity(total_e_entries);
            for i in 0..total_e_entries {
                e_s.push(TestField::from((s_e + 1) as u64 * 10 + (i + 1) as u64));
            }
            E_out_vec.push(e_s);
        }

        let num_zero = num_accums_zero_entries_test_helper(num_svo_rounds);
        let num_infty = num_accums_infty_entries_test_helper(num_svo_rounds);


        let mut accums_zero_new = vec![TestField::zero(); num_zero];
        let mut accums_infty_new = vec![TestField::zero(); num_infty];
        let mut accums_zero_old = vec![TestField::zero(); num_zero];
        let mut accums_infty_old = vec![TestField::zero(); num_infty];

        match num_svo_rounds {
            1 => {
                distribute_tA_to_svo_accumulators::<1, TestField>(&tA_accums, x_out_val, &E_out_vec, &mut accums_zero_new, &mut accums_infty_new);
                distribute_tA_to_svo_accumulators_1(&tA_accums, x_out_val, &E_out_vec, &mut accums_zero_old, &mut accums_infty_old);
            }
            2 => {
                distribute_tA_to_svo_accumulators::<2, TestField>(&tA_accums, x_out_val, &E_out_vec, &mut accums_zero_new, &mut accums_infty_new);
                distribute_tA_to_svo_accumulators_2(&tA_accums, x_out_val, &E_out_vec, &mut accums_zero_old, &mut accums_infty_old);
            }
            3 => {
                distribute_tA_to_svo_accumulators::<3, TestField>(&tA_accums, x_out_val, &E_out_vec, &mut accums_zero_new, &mut accums_infty_new);
                distribute_tA_to_svo_accumulators_3(&tA_accums, x_out_val, &E_out_vec, &mut accums_zero_old, &mut accums_infty_old);
            }
            _ => panic!("Unsupported NUM_SVO_ROUNDS for distribute consistency test"),
        }

        assert_eq!(accums_zero_new, accums_zero_old, "accums_zero mismatch for N={}", num_svo_rounds);
        assert_eq!(accums_infty_new, accums_infty_old, "accums_infty mismatch for N={}", num_svo_rounds);
    }

    #[test]
    fn test_distribute_tA_vs_hardcoded_1() {
        test_distribute_consistency(1);
    }

    #[test]
    fn test_distribute_tA_vs_hardcoded_2() {
        test_distribute_consistency(2);
    }

    #[test]
    fn test_distribute_tA_vs_hardcoded_3() {
        test_distribute_consistency(3);
    }
}