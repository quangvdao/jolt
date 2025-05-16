// Small Value Optimization (SVO) helpers for Spartan first sum-check

use crate::field::{JoltField, OptimizedMulI128};
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
            _ => panic!("Unsupported number of SVO rounds"),
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

        // Extended evaluations (points with at least one 'I')
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
        let az000 = binary_az_evals[0]; let bz000 = binary_bz_evals[0];
        let az001 = binary_az_evals[1]; let bz001 = binary_bz_evals[1];
        let az010 = binary_az_evals[2]; let bz010 = binary_bz_evals[2];
        let az011 = binary_az_evals[3]; let bz011 = binary_bz_evals[3];
        let az100 = binary_az_evals[4]; let bz100 = binary_bz_evals[4];
        let az101 = binary_az_evals[5]; let bz101 = binary_bz_evals[5];
        let az110 = binary_az_evals[6]; let bz110 = binary_bz_evals[6];
        let az111 = binary_az_evals[7]; let bz111 = binary_bz_evals[7];

        // Precompute all first-order extensions (single infinity)
        // These depend only on binary evaluations.
        let az_00I = az001 - az000; let bz_00I = bz001 - bz000;
        let az_01I = az011 - az010; let bz_01I = bz011 - bz010;
        let az_10I = az101 - az100; let bz_10I = bz101 - bz100;
        let az_11I = az111 - az110; let bz_11I = bz111 - bz110;

        let az_0I0 = az010 - az000; let bz_0I0 = bz010 - bz000;
        let az_0I1 = az011 - az001; let bz_0I1 = bz011 - bz001;
        let az_1I0 = az110 - az100; let bz_1I0 = bz110 - bz100;
        let az_1I1 = az111 - az101; let bz_1I1 = bz111 - bz101;

        let az_I00 = az100 - az000; let bz_I00 = bz100 - bz000;
        let az_I01 = az101 - az001; let bz_I01 = bz101 - bz001;
        let az_I10 = az110 - az010; let bz_I10 = bz110 - bz010;
        let az_I11 = az111 - az011; let bz_I11 = bz111 - bz011;

        // Populate temp_tA in lexicographical MSB-first order
        // Z=0, O=1, I=2 for Y_i

        // Point (0,0,I) -> temp_tA[0]
        if az_00I != 0 {
            if bz_00I != 0 {
                let prod = az_00I.checked_mul(bz_00I).expect("Prod overflow");
                // Test `mul_i128_1_optimized`
                temp_tA[0] += e_in_val.mul_i128_1_optimized(prod);
            }
        }

        // Point (0,1,I) -> temp_tA[1]
        if az_01I != 0 {
            if bz_01I != 0 {
                let prod = az_01I.checked_mul(bz_01I).expect("Prod overflow");
                temp_tA[1] += e_in_val.mul_i128(prod);
            }
        }

        // Point (0,I,0) -> temp_tA[2]
        if az_0I0 != 0 {
            if bz_0I0 != 0 {
                let prod = az_0I0.checked_mul(bz_0I0).expect("Prod overflow");
                temp_tA[2] += e_in_val.mul_i128(prod);
            }
        }

        // Point (0,I,1) -> temp_tA[3]
        if az_0I1 != 0 {
            if bz_0I1 != 0 {
                let prod = az_0I1.checked_mul(bz_0I1).expect("Prod overflow");
                temp_tA[3] += e_in_val.mul_i128(prod);
            }
        }

        // Point (0,I,I) -> temp_tA[4]
        let az_0II = az_01I - az_00I;
        let bz_0II = bz_01I - bz_00I; // Need to compute this outside for III term
        if az_0II != 0 {
            if bz_0II != 0 {
                let prod = az_0II.checked_mul(bz_0II).expect("Prod overflow");
                temp_tA[4] += e_in_val.mul_i128(prod);
            }
        }

        // Point (1,0,I) -> temp_tA[5]
        if az_10I != 0 {
            if bz_10I != 0 {
                let prod = az_10I.checked_mul(bz_10I).expect("Prod overflow");
                temp_tA[5] += e_in_val.mul_i128(prod);
            }
        }

        // Point (1,1,I) -> temp_tA[6]
        if az_11I != 0 {
            if bz_11I != 0 {
                let prod = az_11I.checked_mul(bz_11I).expect("Prod overflow");
                temp_tA[6] += e_in_val.mul_i128(prod);
            }
        }

        // Point (1,I,0) -> temp_tA[7]
        if az_1I0 != 0 {
            if bz_1I0 != 0 {
                let prod = az_1I0.checked_mul(bz_1I0).expect("Prod overflow");
                temp_tA[7] += e_in_val.mul_i128(prod);
            }
        }

        // Point (1,I,1) -> temp_tA[8]
        if az_1I1 != 0 {
            if bz_1I1 != 0 {
                let prod = az_1I1.checked_mul(bz_1I1).expect("Prod overflow");
                temp_tA[8] += e_in_val.mul_i128(prod);
            }
        }

        // Point (1,I,I) -> temp_tA[9]
        let az_1II = az_11I - az_10I;
        let bz_1II = bz_11I - bz_10I; // Need to compute this outside for III term
        if az_1II != 0 {
            if bz_1II != 0 {
                let prod = az_1II.checked_mul(bz_1II).expect("Prod overflow");
                temp_tA[9] += e_in_val.mul_i128(prod);
            }
        }

        // Point (I,0,0) -> temp_tA[10]
        if az_I00 != 0 {
            if bz_I00 != 0 {
                let prod = az_I00.checked_mul(bz_I00).expect("Prod overflow");
                temp_tA[10] += e_in_val.mul_i128(prod);
            }
        }

        // Point (I,0,1) -> temp_tA[11]
        if az_I01 != 0 {
            if bz_I01 != 0 {
                let prod = az_I01.checked_mul(bz_I01).expect("Prod overflow");
                temp_tA[11] += e_in_val.mul_i128(prod);
            }
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
        if az_I10 != 0 {
            if bz_I10 != 0 {
                let prod = az_I10.checked_mul(bz_I10).expect("Prod overflow");
                temp_tA[13] += e_in_val.mul_i128(prod);
            }
        }

        // Point (I,1,1) -> temp_tA[14]
        if az_I11 != 0 {
            if bz_I11 != 0 {
                let prod = az_I11.checked_mul(bz_I11).expect("Prod overflow");
                temp_tA[14] += e_in_val.mul_i128(prod);
            }
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
    /// This is the original version with no bound on num_svo_rounds
    #[inline]
    pub fn compute_and_update_tA_inplace<const NUM_SVO_ROUNDS: usize, F: JoltField>(
        ternary_az_evals: &mut [i128], // Size 3^l0, holds Az evals. Initially populated at binary points. Modified in-place.
        ternary_bz_evals: &mut [i128], // Size 3^l0, holds Bz evals. Initially populated at binary points. Modified in-place.
        e_in_val: &F,                  // E_in[x_in] factor for the current x_in
        temp_tA: &mut [F], // Accumulator vector (size 3^l0), updated with E_in * P_ext contribution.
    ) {
        let num_ternary_points = ternary_az_evals.len();
        let expected_ternary_points = 3_usize
            .checked_pow(NUM_SVO_ROUNDS as u32)
            .expect("Ternary points count overflow");
        debug_assert_eq!(num_ternary_points, expected_ternary_points);
        debug_assert_eq!(ternary_bz_evals.len(), num_ternary_points);
        debug_assert_eq!(temp_tA.len(), num_ternary_points);

        if NUM_SVO_ROUNDS == 0 {
            if num_ternary_points > 0 {
                // Should be size 1 if l0=0
                let az0 = ternary_az_evals[0];
                let bz0 = ternary_bz_evals[0];
                if az0 != 0 && bz0 != 0 {
                    // Early check
                    let product_i128 = az0
                        .checked_mul(bz0)
                        .expect("Az0*Bz0 product overflow i128 in SVO base case");
                    temp_tA[0] += e_in_val.mul_i128(product_i128);
                }
            }
            return;
        }

        // --- In-place Extension Phase ---
        for j_dim_idx in 0..NUM_SVO_ROUNDS {
            let base: usize = 3;
            let num_prefix_vars = j_dim_idx;
            let num_suffix_vars = NUM_SVO_ROUNDS - 1 - j_dim_idx;
            let num_prefix_points = base
                .checked_pow(num_prefix_vars as u32)
                .expect("Prefix power overflow");
            let num_suffix_points = base
                .checked_pow(num_suffix_vars as u32)
                .expect("Suffix power overflow");

            for prefix_int in 0..num_prefix_points {
                for suffix_int in 0..num_suffix_points {
                    let mut coords = vec![0usize; NUM_SVO_ROUNDS];
                    let mut current_suffix_int = suffix_int;
                    for i in (j_dim_idx + 1..NUM_SVO_ROUNDS).rev() {
                        coords[i] = current_suffix_int % base;
                        current_suffix_int /= base;
                    }
                    let mut current_prefix_int = prefix_int;
                    for i in (0..j_dim_idx).rev() {
                        coords[i] = current_prefix_int % base;
                        current_prefix_int /= base;
                    }

                    coords[j_dim_idx] = 0;
                    let idx0 = svo_helpers::get_fixed_radix_index(&coords, base, NUM_SVO_ROUNDS);
                    coords[j_dim_idx] = 1;
                    let idx1 = svo_helpers::get_fixed_radix_index(&coords, base, NUM_SVO_ROUNDS);
                    coords[j_dim_idx] = 2;
                    let idx_inf = svo_helpers::get_fixed_radix_index(&coords, base, NUM_SVO_ROUNDS);

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

                let product_p_i128 = az_final
                    .checked_mul(bz_final)
                    .expect("Az_ext*Bz_ext product overflow i128");
                // product_p_i128 cannot be zero here due to the check above

                // Accumulate contribution from this x_in using mul_i128
                *tA_val += e_in_val.mul_i128(product_p_i128);
            });
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
        // (no term A_1(1,I) as it is not needed i.e. it's an eval at 1)
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

    // // Helper: Base indices for flat accumulator arrays (s = 0, 1, 2 is the round number)
    // const BASE_IDX_ZERO_S1: usize = 0; // Accs for s=1, u=0
    // const BASE_IDX_ZERO_S2: usize = 1; // Accs for s=2, u=0 | Offset by num_accs_zero_s1
    // const BASE_IDX_INFTY_S0: usize = 0; // Accs for s=0, u=I
    // const BASE_IDX_INFTY_S1: usize = 1; // Accs for s=1, u=I | Offset by num_accs_infty_s0
    // const BASE_IDX_INFTY_S2: usize = 4; // Accs for s=2, u=I | Offset by num_accs_infty_s0 + num_accs_infty_s1

    // // Pre-calculate E_out values
    // // For s=0 (Y2 is u), num_y_suffix_vars = 2 (Y0, Y1)
    // let E0_y00 = E_out_vec[0][(x_out_val << 2) | 0b00]; // y_suffix=(0,0)
    // let E0_y01 = E_out_vec[0][(x_out_val << 2) | 0b01]; // y_suffix=(0,1)
    // let E0_y10 = E_out_vec[0][(x_out_val << 2) | 0b10]; // y_suffix=(1,0)
    // let E0_y11 = E_out_vec[0][(x_out_val << 2) | 0b11]; // y_suffix=(1,1)

    // // For s=1 (Y1 is u, Y2 is v_0), num_y_suffix_vars = 1 (Y0)
    // let E1_y0 = E_out_vec[1][(x_out_val << 1) | 0];    // y_suffix=(0)
    // let E1_y1 = E_out_vec[1][(x_out_val << 1) | 1];    // y_suffix=(1)

    // // For s=2 (Y0 is u, Y1 is v_1, Y2 is v_0), num_y_suffix_vars = 0 (empty y_suffix)
    // let E2_yempty = E_out_vec[2][x_out_val];

    // Y_ext=(0,0,I) -> tA[0]. Worked out example for 3 case.
    // Contributes to A_0(I) (accums_infty[BASE_IDX_INFTY_S0 + 0]) via E0_y00 * tA[0]
    // Contributes to A_1(0,I) (accums_zero[BASE_IDX_ZERO_S1 + 0]) via E1_y0 * tA[0]
    // Contributes to A_2(0,0,I) (accums_zero[BASE_IDX_ZERO_S2 + 0]) via E2_yempty * tA[0]

    /// Hardcoded version for `num_svo_rounds == 3`
    /// TODO: refactor for correct distribution
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

        use SVOEvalPoint::{Zero, One, Infinity}; 

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
        let e1_suf0 = E_out_vec[1][(x_out_val << 1) | 0];    // y_suffix_eff=(Y0_c=0)
        let e1_suf1 = E_out_vec[1][(x_out_val << 1) | 1];    // y_suffix_eff=(Y0_c=1)
        
        // E_out_vec[2] (s_code=2 in E_out_vec): Y0_c is u_eff for E. y_suffix_eff=(). Index x_out_val
        let e2_sufempty = E_out_vec[2][x_out_val];

        // Y_EXT_CODE_MAP[tA_idx] gives (Y0_c, Y1_c, Y2_c) for tA_accums[tA_idx]
        // Y0_c is MSB, Y2_c is LSB for code variables. This order matches compute_and_update_tA_inplace_3.
        const Y_EXT_CODE_MAP: [(SVOEvalPoint, SVOEvalPoint, SVOEvalPoint); 19] = [
            (Zero, Zero, Infinity), (Zero, One, Infinity), (Zero, Infinity, Zero), (Zero, Infinity, One), (Zero, Infinity, Infinity),
            (One, Zero, Infinity), (One, One, Infinity), (One, Infinity, Zero), (One, Infinity, One), (One, Infinity, Infinity),
            (Infinity, Zero, Zero), (Infinity, Zero, One), (Infinity, Zero, Infinity), (Infinity, One, Zero), (Infinity, One, One),
            (Infinity, One, Infinity), (Infinity, Infinity, Zero), (Infinity, Infinity, One), (Infinity, Infinity, Infinity),
        ];

        for i in 0..19 {
            let current_tA = tA_accums[i];
            // No !current_tA.is_zero() check, as requested

            let (y0_c, y1_c, y2_c) = Y_EXT_CODE_MAP[i]; // (MSB, Mid, LSB) from code's perspective

            // --- Contributions to Paper Round s_p=0 Accumulators (u_paper = Y2_c) ---
            // E-factor uses E_out_vec[0] (where Y2_c is u_eff), E_suffix is (Y0_c, Y1_c)
            if y2_c == Infinity { // u_paper = I
                if y0_c != Infinity && y1_c != Infinity { // Suffix (Y0_c,Y1_c) for E_out_vec[0] must be binary
                    let e_val = match (y0_c, y1_c) {
                        (Zero, Zero) => e0_suf00,
                        (Zero, One)  => e0_suf01,
                        (One, Zero)  => e0_suf10,
                        (One, One)   => e0_suf11,
                        _ => unreachable!(), // Should be covered by binary check above
                    };
                    accums_infty[ACCUM_IDX_A0_I] += current_tA * e_val;
                }
            }
            // No A_0(0) slots defined in the provided consts for accums_zero.

            // --- Contributions to Paper Round s_p=1 Accumulators (u_paper = Y1_c, v0_paper = Y2_c) ---
            // E-factor uses E_out_vec[1] (where Y1_c is u_eff), E_suffix is (Y0_c)
            if y0_c != Infinity { // Suffix Y0_c for E_out_vec[1] must be binary
                let e1_val = if y0_c == Zero { e1_suf0 } else { e1_suf1 }; // y0_c is One or Zero here
                
                if y1_c == Infinity { // u_paper = I
                    match y2_c { // v0_paper = Y2_c
                        Zero     => { accums_infty[ACCUM_IDX_A1_0_I] += current_tA * e1_val; }
                        One      => { accums_infty[ACCUM_IDX_A1_1_I] += current_tA * e1_val; }
                        Infinity => { accums_infty[ACCUM_IDX_A1_I_I] += current_tA * e1_val; }
                    }
                } else if y1_c == Zero { // u_paper = 0
                    if y2_c == Infinity { // v0_paper = I, for A_1(I,0)
                        accums_zero[ACCUM_IDX_A1_I_0] += current_tA * e1_val;
                    }
                }
            }

            // --- Contributions to Paper Round s_p=2 Accumulators (u_paper = Y0_c, v_paper = (Y1_c,Y2_c)) ---
            // E-factor uses E_out_vec[2] (where Y0_c is u_eff), E_suffix is empty
            let e2_val = e2_sufempty;
            if y0_c == Infinity { // u_paper = I
                 match (y1_c, y2_c) { // v_paper = (Y1_c, Y2_c)
                    (Zero, Zero)         => { accums_infty[ACCUM_IDX_A2_00_I] += current_tA * e2_val; }
                    (Zero, One)          => { accums_infty[ACCUM_IDX_A2_01_I] += current_tA * e2_val; }
                    (Zero, Infinity)     => { accums_infty[ACCUM_IDX_A2_0I_I] += current_tA * e2_val; }
                    (One, Zero)          => { accums_infty[ACCUM_IDX_A2_10_I] += current_tA * e2_val; }
                    (One, One)           => { accums_infty[ACCUM_IDX_A2_11_I] += current_tA * e2_val; }
                    (One, Infinity)      => { accums_infty[ACCUM_IDX_A2_1I_I] += current_tA * e2_val; }
                    (Infinity, Zero)     => { accums_infty[ACCUM_IDX_A2_I0_I] += current_tA * e2_val; }
                    (Infinity, One)      => { accums_infty[ACCUM_IDX_A2_I1_I] += current_tA * e2_val; }
                    (Infinity, Infinity) => { accums_infty[ACCUM_IDX_A2_II_I] += current_tA * e2_val; }
                 }
            } else if y0_c == Zero { // u_paper = 0
                match (y1_c, y2_c) { // v_paper = (Y1_c, Y2_c)
                    // Only specific v_paper configs for A_2(v,0) are stored, based on ACCUM_IDX constants
                    (Zero, Infinity)     => { accums_zero[ACCUM_IDX_A2_0I_0] += current_tA * e2_val; }
                    (One, Infinity)      => { accums_zero[ACCUM_IDX_A2_1I_0] += current_tA * e2_val; }
                    (Infinity, Zero)     => { accums_zero[ACCUM_IDX_A2_I0_0] += current_tA * e2_val; }
                    (Infinity, One)      => { accums_zero[ACCUM_IDX_A2_I1_0] += current_tA * e2_val; }
                    (Infinity, Infinity) => { accums_zero[ACCUM_IDX_A2_II_0] += current_tA * e2_val; }
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
        num_ternary_points: usize,
        E_out_vec: &[Vec<F>],
        all_idx_mapping_results: &[Vec<(usize, Vec<SVOEvalPoint>, SVOEvalPoint, usize, usize)>], // Updated tuple
        task_svo_accs: &mut Vec<(Vec<F>, Vec<F>)>, // Vec over rounds, each tuple is (acc_for_u_Z, acc_for_u_I)
    ) {
        if NUM_SVO_ROUNDS == 0 {
            return;
        }
        debug_assert_eq!(tA_accums.len(), num_ternary_points);
        debug_assert_eq!(all_idx_mapping_results.len(), num_ternary_points);
        debug_assert_eq!(task_svo_accs.len(), NUM_SVO_ROUNDS);
        debug_assert_eq!(E_out_vec.len(), NUM_SVO_ROUNDS);

        for beta_idx in 0..num_ternary_points {
            let tA_for_this_beta = tA_accums[beta_idx];
            if tA_for_this_beta.is_zero() {
                continue;
            }

            let relevant_mappings = &all_idx_mapping_results[beta_idx];

            for (round_s, v_config_vec, u_eval_point, y_suffix_as_int, num_y_suffix_vars) in
                relevant_mappings.iter().cloned()
            {
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
                            task_svo_accs[round_s].0[v_config_idx] +=
                                E_out_s_val * tA_for_this_beta;
                        }
                        SVOEvalPoint::Infinity => {
                            task_svo_accs[round_s].1[v_config_idx] +=
                                E_out_s_val * tA_for_this_beta;
                        }
                        SVOEvalPoint::One => {
                            panic!("SVO accumulators only store u=0/inf");
                        }
                    }
                }
            }
        }
    }

    // Computes Teq factor for E_out_s
    // This function now uses the precomputed E_out_s evaluations from eq_poly.E1.
    // Corresponds to accessing E_out,i[y, x_out] in Algorithm 6 (proc:precompute-algo-6, Line 14).
    #[inline(always)]
    pub fn get_E_out_s_val<F: JoltField>(
        E_out_s_evals: &[F],      // Precomputed evaluations E_out,i for round i=s
        num_y_suffix_vars: usize, // Number of variables in y_suffix (length of y in E_out,i[y, x_out])
        y_suffix_as_int: usize,   // Integer representation of y_suffix (LSB)
        x_out_val: usize,         // Integer assignment for x_out variables (MSB)
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
    ) -> Vec<(usize, Vec<SVOEvalPoint>, SVOEvalPoint, usize, usize)> {
        // Added y_suffix_as_int, num_y_suffix_vars
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
            debug_assert!(
                point_coords[i] < base,
                "Coord {} out of bounds for base {}",
                point_coords[i],
                base
            );
            index += point_coords[i] * stride;
            if i > 0 {
                // Avoid overflow on last stride calculation
                // Use checked_mul for safety
                stride = stride
                    .checked_mul(base)
                    .expect("Base power overflow computing stride");
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
                stride = stride
                    .checked_mul(3)
                    .expect("Stride overflow in binary_to_ternary_index");
            }
            current_binary_val /= 2;
        }
        if current_binary_val != 0 {
            // This indicates the binary_idx was too large for num_svo_rounds
            panic!(
                "binary_idx {} too large for {} SVO rounds",
                binary_idx, num_svo_rounds
            );
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
    pub fn get_svo_prefix_extended_from_idx(
        beta_idx: usize,
        num_svo_rounds: usize,
    ) -> Vec<SVOEvalPoint> {
        if num_svo_rounds == 0 {
            return vec![];
        }
        let mut temp_beta_coords = vec![0usize; num_svo_rounds];
        let mut temp_beta_int = beta_idx;
        for i in (0..num_svo_rounds).rev() {
            // MSB first for coords array from LSB of int
            temp_beta_coords[i] = temp_beta_int % 3;
            temp_beta_int /= 3;
        }
        temp_beta_coords
            .iter()
            .map(|&coord| match coord {
                0 => SVOEvalPoint::Zero,
                1 => SVOEvalPoint::One,
                2 => SVOEvalPoint::Infinity,
                _ => unreachable!("Invalid coordinate in beta_idx to SVOEvalPoint conversion"),
            })
            .collect::<Vec<_>>()
    }

    // Precomputes idx_mapping results for all possible beta values.
    #[inline]
    pub fn precompute_all_idx_mappings(
        num_svo_rounds: usize,
        num_ternary_points: usize,
    ) -> Vec<Vec<(usize, Vec<SVOEvalPoint>, SVOEvalPoint, usize, usize)>> {
        // Updated tuple
        if num_svo_rounds == 0 {
            return vec![vec![]];
        }
        (0..num_ternary_points)
            .map(|beta_idx| {
                let svo_prefix = get_svo_prefix_extended_from_idx(beta_idx, num_svo_rounds);
                idx_mapping(&svo_prefix, num_svo_rounds) // idx_mapping now returns the richer tuple
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::svo_helpers::*;
    use ark_bn254::Fr as TestField;
    use ark_ff::Zero;
    use SVOEvalPoint::*;
    // use crate::poly::spartan_interleaved_poly::{num_accums_eval_zero, num_accums_eval_infty};

    /// Tests the `get_svo_prefix_extended_from_idx` function.
    /// This function converts a base-3 integer index into a vector of `SVOEvalPoint`s.
    /// It covers cases for 0, 1, 2, and 3 SVO rounds, ensuring correct MSB-first
    /// conversion from the ternary representation of the index.
    #[test]
    fn test_get_svo_prefix_extended_from_idx() {
        // num_svo_rounds = 0
        assert_eq!(get_svo_prefix_extended_from_idx(0, 0), vec![]);

        // num_svo_rounds = 1
        assert_eq!(
            get_svo_prefix_extended_from_idx(0, 1),
            vec![SVOEvalPoint::Zero]
        );
        assert_eq!(
            get_svo_prefix_extended_from_idx(1, 1),
            vec![SVOEvalPoint::One]
        );
        assert_eq!(
            get_svo_prefix_extended_from_idx(2, 1),
            vec![SVOEvalPoint::Infinity]
        );

        // num_svo_rounds = 2. Indices are in base 3, MSB first in output vector.
        // 0 = 00_3
        assert_eq!(
            get_svo_prefix_extended_from_idx(0, 2),
            vec![SVOEvalPoint::Zero, SVOEvalPoint::Zero]
        );
        // 1 = 01_3
        assert_eq!(
            get_svo_prefix_extended_from_idx(1, 2),
            vec![SVOEvalPoint::Zero, SVOEvalPoint::One]
        );
        // 2 = 02_3
        assert_eq!(
            get_svo_prefix_extended_from_idx(2, 2),
            vec![SVOEvalPoint::Zero, SVOEvalPoint::Infinity]
        );
        // 3 = 10_3
        assert_eq!(
            get_svo_prefix_extended_from_idx(3, 2),
            vec![SVOEvalPoint::One, SVOEvalPoint::Zero]
        );
        // 4 = 11_3
        assert_eq!(
            get_svo_prefix_extended_from_idx(4, 2),
            vec![SVOEvalPoint::One, SVOEvalPoint::One]
        );
        // 8 = 22_3
        assert_eq!(
            get_svo_prefix_extended_from_idx(8, 2),
            vec![SVOEvalPoint::Infinity, SVOEvalPoint::Infinity]
        );

        // num_svo_rounds = 3
        // 0 = 000_3
        assert_eq!(
            get_svo_prefix_extended_from_idx(0, 3),
            vec![SVOEvalPoint::Zero, SVOEvalPoint::Zero, SVOEvalPoint::Zero]
        );
        // 13 = 111_3  (1*9 + 1*3 + 1*1)
        assert_eq!(
            get_svo_prefix_extended_from_idx(13, 3),
            vec![SVOEvalPoint::One, SVOEvalPoint::One, SVOEvalPoint::One]
        );
        // 26 = 222_3 (2*9 + 2*3 + 2*1)
        assert_eq!(
            get_svo_prefix_extended_from_idx(26, 3),
            vec![
                SVOEvalPoint::Infinity,
                SVOEvalPoint::Infinity,
                SVOEvalPoint::Infinity
            ]
        );
        // 5 = 012_3 (0*9 + 1*3 + 2*1)
        assert_eq!(
            get_svo_prefix_extended_from_idx(5, 3),
            vec![
                SVOEvalPoint::Zero,
                SVOEvalPoint::One,
                SVOEvalPoint::Infinity
            ]
        );
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
        assert_eq!(
            map_v_config_to_idx(&[SVOEvalPoint::Zero, SVOEvalPoint::Zero], 1),
            0
        ); // 0*3^0 + 0*3^1
        assert_eq!(
            map_v_config_to_idx(&[SVOEvalPoint::One, SVOEvalPoint::Zero], 1),
            1
        ); // 1*3^0 + 0*3^1
        assert_eq!(
            map_v_config_to_idx(&[SVOEvalPoint::Zero, SVOEvalPoint::One], 1),
            3
        ); // 0*3^0 + 1*3^1
        assert_eq!(
            map_v_config_to_idx(&[SVOEvalPoint::Infinity, SVOEvalPoint::Infinity], 1),
            8
        ); // 2*3^0 + 2*3^1 = 2+6

        assert_eq!(
            map_v_config_to_idx(
                &[SVOEvalPoint::One, SVOEvalPoint::One, SVOEvalPoint::One],
                2
            ),
            13
        ); // 1*1 + 1*3 + 1*9
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
        assert_eq!(get_fixed_radix_index(&[0, 0, 0], 2, 3), 0); // 0*1+0*2+0*4
        assert_eq!(get_fixed_radix_index(&[0, 0, 1], 2, 3), 1); // 1*1+0*2+0*4
        assert_eq!(get_fixed_radix_index(&[0, 1, 0], 2, 3), 2); // 0*1+1*2+0*4
        assert_eq!(get_fixed_radix_index(&[1, 1, 1], 2, 3), 7); // 1*1+1*2+1*4
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
        assert_eq!(binary_to_ternary_index(0b000, 3), 0); // 0
        assert_eq!(binary_to_ternary_index(0b001, 3), 1); // 1
        assert_eq!(binary_to_ternary_index(0b010, 3), 3); // 3
        assert_eq!(binary_to_ternary_index(0b011, 3), 4); // 4
        assert_eq!(binary_to_ternary_index(0b100, 3), 9); // 9
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
        assert_eq!(
            precompute_binary_to_ternary_indices(3),
            vec![0, 1, 3, 4, 9, 10, 12, 13]
        );
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
        assert_eq!(
            idx_mapping(&[Infinity], 1),
            vec![(0, vec![], Infinity, 0, 0)]
        );

        // num_svo_rounds = 2
        // beta = [Z, Z]
        let res_zz = idx_mapping(&[Zero, Zero], 2);
        assert_eq!(res_zz.len(), 2);
        assert!(res_zz.contains(&(0, vec![], Zero, 0, 1))); // s=0, v=[], u=Z, y=[Z] -> y_val=0, y_len=1
        assert!(res_zz.contains(&(1, vec![Zero], Zero, 0, 0))); // s=1, v=[Z], u=Z, y=[] -> y_val=0, y_len=0

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
        let res_zzz = idx_mapping(&[Zero, Zero, Zero], 3);
        assert_eq!(res_zzz.len(), 3);
        assert!(res_zzz.contains(&(0, vec![], Zero, 0, 2))); // y=[Z,Z]
        assert!(res_zzz.contains(&(1, vec![Zero], Zero, 0, 1))); // y=[Z]
        assert!(res_zzz.contains(&(2, vec![Zero, Zero], Zero, 0, 0))); // y=[]

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
            assert_eq!(
                map_2[beta_idx],
                idx_mapping(&svo_prefix, 2),
                "Mismatch for beta_idx {}",
                beta_idx
            );
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

        compute_and_update_tA_inplace::<0, TestField>(
            &mut az_evals_0,
            &mut bz_evals_0,
            &e_in_val_0,
            &mut temp_tA_0,
        );

        // Expected: e_in_val * az * bz = 2 * 5 * 3 = 30
        assert_eq!(temp_tA_0[0], TestField::from(30u64));

        // Test num_svo_rounds = 1
        // For l0=1, we have 3 points (0,1,I) and need to test extension
        let mut _az_evals_1 = vec![2i128, 4i128, 0i128]; // Only 0 and 1 points initially populated
        let mut _bz_evals_1 = vec![3i128, 5i128, 0i128];
        let mut temp_tA_1 = vec![TestField::zero(); 3];
        let e_in_val_1 = TestField::from(1u64);

        // Manually populate ternary arrays for the generic function based on binary inputs.
        // binary_az_evals_1 = [2, 4], binary_bz_evals_1 = [3, 5]
        // ternary_az_evals_1 needs to be [az(0), az(1), 0] initially for extension
        // ternary_bz_evals_1 needs to be [bz(0), bz(1), 0] initially for extension
        // index for 0 in base-3 (1 var) is 0. index for 1 in base-3 (1 var) is 1.
        let mut ternary_az_for_generic_1 = vec![0i128; 3];
        let mut ternary_bz_for_generic_1 = vec![0i128; 3];
        ternary_az_for_generic_1[binary_to_ternary_index(0, 1)] = 2i128; // az(0)
        ternary_az_for_generic_1[binary_to_ternary_index(1, 1)] = 4i128; // az(1)
        ternary_bz_for_generic_1[binary_to_ternary_index(0, 1)] = 3i128; // bz(0)
        ternary_bz_for_generic_1[binary_to_ternary_index(1, 1)] = 5i128; // bz(1)


        compute_and_update_tA_inplace::<1, TestField>(
            &mut ternary_az_for_generic_1,
            &mut ternary_bz_for_generic_1,
            &e_in_val_1,
            &mut temp_tA_1,
        );

        // After extension:
        // ternary_az_for_generic_1 should be [2, 4, 2] (I value is 4-2=2)
        // ternary_bz_for_generic_1 should be [3, 5, 2] (I value is 5-3=2)
        assert_eq!(ternary_az_for_generic_1, vec![2i128, 4i128, 2i128]);
        assert_eq!(ternary_bz_for_generic_1, vec![3i128, 5i128, 2i128]);

        // Expected: e_in_val * az * bz
        assert_eq!(temp_tA_1[0], TestField::from(6u64)); // 1 * 2 * 3 = 6
        assert_eq!(temp_tA_1[1], TestField::from(20u64)); // 1 * 4 * 5 = 20
        assert_eq!(temp_tA_1[2], TestField::from(4u64)); // 1 * 2 * 2 = 4
    }

    // Test for distribute_tA_to_svo_accumulators
    #[test]
    fn test_distribute_tA_to_svo_accumulators() {
        // Test scenario: l0=2, round_s in {0,1}
        // Num ternary points = 3^2 = 9
        let num_svo_rounds = 2;
        let num_ternary_points = 9;
        let x_out_val = 0; // Simple case with x_out=0

        // Prepare tA_accums - Only populate a few indices to test specific paths
        // Values are P_ext(Y_ext) * E_in. For simplicity, assume E_in is 1 for these points.
        let mut tA_accums = vec![TestField::zero(); num_ternary_points];
        // Y_ext = [Z,Z] (beta_idx 0 via get_fixed_radix_index(&[0,0],3,2)). task_tA = 2.
        tA_accums[get_fixed_radix_index(&[0,0],3,num_svo_rounds)] = TestField::from(2u64);
        // Y_ext = [Z,I] (beta_idx 2 via get_fixed_radix_index(&[0,2],3,2)). task_tA = 5.
        tA_accums[get_fixed_radix_index(&[0,2],3,num_svo_rounds)] = TestField::from(5u64);
        // Y_ext = [O,O] (beta_idx 4 via get_fixed_radix_index(&[1,1],3,2)). task_tA = 3.
        tA_accums[get_fixed_radix_index(&[1,1],3,num_svo_rounds)] = TestField::from(3u64);
        // Y_ext = [I,Z] (beta_idx 6 via get_fixed_radix_index(&[2,0],3,2)). task_tA = 7.
        tA_accums[get_fixed_radix_index(&[2,0],3,num_svo_rounds)] = TestField::from(7u64);
        // Y_ext = [I,O] (beta_idx 7 via get_fixed_radix_index(&[2,1],3,2)). task_tA = 8.
        tA_accums[get_fixed_radix_index(&[2,1],3,num_svo_rounds)] = TestField::from(8u64);
        // Y_ext = [O,I] (beta_idx 5 via get_fixed_radix_index(&[1,2],3,2)). task_tA = 6.
        tA_accums[get_fixed_radix_index(&[1,2],3,num_svo_rounds)] = TestField::from(6u64);
        // Y_ext = [I,I] (beta_idx 8 via get_fixed_radix_index(&[2,2],3,2)). task_tA = 9.
        tA_accums[get_fixed_radix_index(&[2,2],3,num_svo_rounds)] = TestField::from(9u64);


        // Prepare E_out_vec 
        let mut E_out_vec = Vec::new();
        // For round_s=0: num_y_suffix_vars=1 (Y1), num_x_out_vars=1 (X0). Table length 2^(1+1) = 4. Index (X0 << 1 | Y1)
        let E_out_s0 = vec![
            TestField::from(1u64), // E[X0=0,Y1=0]
            TestField::from(2u64), // E[X0=0,Y1=1]
            TestField::from(3u64), // E[X0=1,Y1=0]
            TestField::from(4u64), // E[X0=1,Y1=1]
        ];
        // For round_s=1: num_y_suffix_vars=0, num_x_out_vars=1 (X0). Table length 2^(0+1) = 2. Index (X0)
        let E_out_s1 = vec![
            TestField::from(5u64), // E[X0=0]
            TestField::from(6u64), // E[X0=1]
        ];
        E_out_vec.push(E_out_s0);
        E_out_vec.push(E_out_s1);

        let all_idx_mapping_results =
            precompute_all_idx_mappings(num_svo_rounds, num_ternary_points);

        let mut task_svo_accs = Vec::new();
        // Round 0: s=0, num_v_configs = 3^0 = 1
        task_svo_accs.push((
            vec![TestField::zero()], // u=Z, v=[] 
            vec![TestField::zero()], // u=I, v=[] 
        ));
        // Round 1: s=1, num_v_configs = 3^1 = 3 
        task_svo_accs.push((
            vec![TestField::zero(); 3], // u=Z, v in {Z,O,I}
            vec![TestField::zero(); 3], // u=I, v in {Z,O,I}
        ));

        distribute_tA_to_svo_accumulators::<2, TestField>(
            &tA_accums,
            x_out_val,
            num_ternary_points,
            &E_out_vec,
            &all_idx_mapping_results,
            &mut task_svo_accs,
        );

        // Expected values based on our knowledge of the mappings and values
        // For beta_idx = 0 ([Z,Z]), task_tA = 2:
        // - Mappings from all_idx_mapping_results[0]:
        //   - (0, [], Z, 0, 1) -> round_s=0, v_config=[], u=Z, y_suffix=0 ([Z]), num_y_suffix=1
        //     This is skipped by `if is_v_config_binary && u_eval_point == SVOEvalPoint::Zero`
        //     So task_svo_accs[0].0[0] (A_0([],Z)) should NOT get this contribution.
        //   - (1, [Z], Z, 0, 0) -> round_s=1, v_config=[Z], u=Z, y_suffix=0, num_y_suffix=0
        //     E_out_1_val = E_out_vec[1][(x_out_val << 0) | 0] = E_out_vec[1][0] = 5
        //     v_config_idx = map_v_config_to_idx(&[Z], 1) = 0
        //     task_svo_accs[1].0[0] += 5 * 2 = 10
        assert_eq!(task_svo_accs[0].0[0], TestField::zero(), "A_0(v=[], u=Z)");

        // task_svo_accs[0].1[0] is A_0(v=[], u=I)
        //   - Y_ext=[I,Z] (beta 6, tA=7) -> maps to (s=0,v=[],u=I,y_sfx=[Z]->0). E_out_s0[(0<<1)|0]=E_out_s0[0]=1. Contr: 7*1=7.
        //   - Y_ext=[I,O] (beta 7, tA=8) -> maps to (s=0,v=[],u=I,y_sfx=[O]->1). E_out_s0[(0<<1)|1]=E_out_s0[1]=2. Contr: 8*2=16.
        //   - Y_ext=[I,I] (beta 8, tA=9) -> maps to (s=0,v=[],u=I,y_sfx=[I]). Skipped by idx_mapping (y_sfx not binary).
        assert_eq!(task_svo_accs[0].1[0], TestField::from(7u64 + 16u64), "A_0(v=[], u=I)");

        // task_svo_accs[1] is for s=1 (Y1 is current variable u, Y0 is in v_config)
        // task_svo_accs[1].0[map_v_config_to_idx(&[Z],1)=0] is A_1(v=[Z], u=Z)
        //   - Y_ext=[Z,Z] (beta 0, tA=2) -> maps to (s=1,v=[Z],u=Z,y_sfx=[]). v=[Z] is binary, u=Z. Skipped.
        assert_eq!(task_svo_accs[1].0[0], TestField::zero(), "A_1(v=[Z], u=Z)");

        // task_svo_accs[1].0[map_v_config_to_idx(&[O],1)=1] is A_1(v=[O], u=Z)
        //   - Y_ext=[O,Z] (beta 3, tA=0) -> maps to (s=1,v=[O],u=Z,y_sfx=[]). v=[O] is binary, u=Z. Skipped.
        assert_eq!(task_svo_accs[1].0[1], TestField::zero(), "A_1(v=[O], u=Z)");

        // task_svo_accs[1].0[map_v_config_to_idx(&[I],1)=2] is A_1(v=[I], u=Z)
        //   - Y_ext=[I,Z] (beta 6, tA=7) -> maps to (s=1,v=[I],u=Z,y_sfx=[]). v=[I] not binary. E_out_s1[0]=5. Contr: 7*5=35.
        assert_eq!(task_svo_accs[1].0[2], TestField::from(35u64), "A_1(v=[I], u=Z)");

        // task_svo_accs[1].1[map_v_config_to_idx(&[Z],1)=0] is A_1(v=[Z], u=I)
        //   - Y_ext=[Z,I] (beta 2, tA=5) -> maps to (s=1,v=[Z],u=I,y_sfx=[]). E_out_s1[0]=5. Contr: 5*5=25.
        assert_eq!(task_svo_accs[1].1[0], TestField::from(25u64), "A_1(v=[Z], u=I)");

        // task_svo_accs[1].1[map_v_config_to_idx(&[O],1)=1] is A_1(v=[O], u=I)
        //   - Y_ext=[O,I] (beta 5, tA=6) -> maps to (s=1,v=[O],u=I,y_sfx=[]). E_out_s1[0]=5. Contr: 6*5=30.
        assert_eq!(task_svo_accs[1].1[1], TestField::from(30u64), "A_1(v=[O], u=I)");

        // task_svo_accs[1].1[map_v_config_to_idx(&[I],1)=2] is A_1(v=[I], u=I)
        //   - Y_ext=[I,I] (beta 8, tA=9) -> maps to (s=1,v=[I],u=I,y_sfx=[]). E_out_s1[0]=5. Contr: 9*5=45.
        assert_eq!(task_svo_accs[1].1[2], TestField::from(45u64), "A_1(v=[I], u=I)");
    }

    fn _initialize_ternary_from_binary<const NUM_SVO_ROUNDS: usize>(
        binary_evals: &[i128],
        ternary_evals: &mut [i128],
    ) {
        let num_binary_points = 1 << NUM_SVO_ROUNDS;
        assert_eq!(binary_evals.len(), num_binary_points);
        assert_eq!(ternary_evals.len(), 3_usize.pow(NUM_SVO_ROUNDS as u32));
        for i in 0..ternary_evals.len() {
            ternary_evals[i] = 0; // Initialize all to 0
        }
        for bin_idx in 0..num_binary_points {
            let tern_idx = binary_to_ternary_index(bin_idx, NUM_SVO_ROUNDS);
            ternary_evals[tern_idx] = binary_evals[bin_idx];
        }
    }
}
