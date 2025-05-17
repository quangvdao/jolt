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
            _ => compute_and_update_tA_inplace_generic::<NUM_SVO_ROUNDS, F>(binary_az_evals, binary_bz_evals, e_in_val, temp_tA),
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

        // Precompute all first-order extensions (single infinity)
        // These depend only on binary evaluations.
        let az_00I = az001 - az000;
        let bz_00I = bz001 - bz000;
        let az_01I = az011 - az010;
        let bz_01I = bz011 - bz010;
        let az_10I = az101 - az100;
        let bz_10I = bz101 - bz100;
        let az_11I = az111 - az110;
        let bz_11I = bz111 - bz110;

        let az_0I0 = az010 - az000;
        let bz_0I0 = bz010 - bz000;
        let az_0I1 = az011 - az001;
        let bz_0I1 = bz011 - bz001;
        let az_1I0 = az110 - az100;
        let bz_1I0 = bz110 - bz100;
        let az_1I1 = az111 - az101;
        let bz_1I1 = bz111 - bz101;

        let az_I00 = az100 - az000;
        let bz_I00 = bz100 - bz000;
        let az_I01 = az101 - az001;
        let bz_I01 = bz101 - bz001;
        let az_I10 = az110 - az010;
        let bz_I10 = bz110 - bz010;
        let az_I11 = az111 - az011;
        let bz_I11 = bz111 - bz011;

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
        binary_az_evals: &mut [i128],
        binary_bz_evals: &mut [i128],
        e_in_val: &F,
        temp_tA: &mut [F],
    ) {
        todo!()
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

            let (y0_c, y1_c, y2_c) = Y_EXT_CODE_MAP[i]; // (MSB, Mid, LSB) from code's perspective

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
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::svo_helpers::*;
    use ark_bn254::Fr as TestField;
    use ark_ff::Zero;
    use SVOEvalPoint::*;
    // use crate::poly::spartan_interleaved_poly::{num_accums_eval_zero, num_accums_eval_infty};

    // TODO: add tests after refactoring the logic
}