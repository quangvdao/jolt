// Small Value Optimization (SVO) helpers for Spartan first sum-check

use crate::field::JoltField;
use crate::poly::multilinear_polynomial::MultilinearPolynomial;
use crate::r1cs::builder::{Constraint, OffsetEqConstraint, eval_offset_lc_i64};
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
    pub fn evaluate_Az_Bz_for_r1cs_row_binary<F: JoltField>(
        uniform_constraints: &[Constraint],
        cross_step_constraints: &[OffsetEqConstraint],
        flattened_polynomials: &[&MultilinearPolynomial<F>],
        current_step_idx: usize,
        constraint_idx_within_step: usize, // Full constraint index including SVO bits (all binary)
        num_uniform_constraints: usize,
        num_total_steps: usize,
    ) -> (i64, i64) {
        // Returns (az_coeff_i64, bz_coeff_i64)
        let mut az_i64 = 0i64;
        let mut bz_i64 = 0i64;

        if constraint_idx_within_step < num_uniform_constraints {
            let constraint = &uniform_constraints[constraint_idx_within_step];
            if !constraint.a.terms().is_empty() {
                az_i64 = constraint
                    .a
                    .evaluate_row_i64(flattened_polynomials, current_step_idx);
            }
            if !constraint.b.terms().is_empty() {
                bz_i64 = constraint
                    .b
                    .evaluate_row_i64(flattened_polynomials, current_step_idx);
            }
        } else if constraint_idx_within_step < num_uniform_constraints + cross_step_constraints.len() {
            let cross_step_constraint_idx = constraint_idx_within_step - num_uniform_constraints;
            let constraint = &cross_step_constraints[cross_step_constraint_idx];
            let next_step_index_opt = if current_step_idx + 1 < num_total_steps {
                Some(current_step_idx + 1)
            } else {
                None
            };

            let eq_a_eval = eval_offset_lc_i64(
                &constraint.a,
                flattened_polynomials,
                current_step_idx,
                next_step_index_opt,
            );
            let eq_b_eval = eval_offset_lc_i64(
                &constraint.b,
                flattened_polynomials,
                current_step_idx,
                next_step_index_opt,
            );
            let temp_az = eq_a_eval - eq_b_eval;

            if !temp_az.is_zero() {
                az_i64 = temp_az;
                // Optional: Assert cond is zero:
                // let cond_eval = eval_offset_lc_i64(&constraint.cond, flattened_polynomials, current_step_idx, next_step_index_opt);
                // assert_eq!(cond_eval, 0, "Cross-step constraint violated");
            } else {
                bz_i64 = eval_offset_lc_i64(
                    &constraint.cond,
                    flattened_polynomials,
                    current_step_idx,
                    next_step_index_opt,
                );
            }
        }
        (az_i64, bz_i64)
    }

    // Computes Teq factor for E_out_s
    // This function now uses the precomputed E_out_s evaluations from eq_poly.E1.
    // Corresponds to accessing E_out,i[y, x_out] in Algorithm 6 (proc:precompute-algo-6, Line 14).
    #[inline]
    pub fn get_E_out_s_val<F: JoltField>(
        E_out_s_evals: &[F],              // Precomputed evaluations E_out,i for round i=s
        num_y_suffix_vars: usize,         // Number of variables in y_suffix (length of y in E_out,i[y, x_out])
        #[cfg(test)]
        num_x_out_vars: usize,            // Number of variables in x_out (length of x_out in E_out,i[y, x_out])
        y_suffix_as_int: usize,           // Integer representation of y_suffix (LSB)
        x_out_val: usize,                 // Integer assignment for x_out variables (MSB)
    ) -> F {
        // Sanity check for evaluation table length
        #[cfg(test)]
        {
            let expected_len = 1 << (num_y_suffix_vars + num_x_out_vars);
            assert_eq!(E_out_s_evals.len(), expected_len, "E_out_s_evals has unexpected length {}. Expected {}. num_y_suffix_vars = {}, num_x_out_vars = {}", E_out_s_evals.len(), expected_len, num_y_suffix_vars, num_x_out_vars);
        }

        // y_suffix_as_int is LSB, x_out_val is MSB. Combine to form the index into E_out,i.
        let combined_idx = (x_out_val << num_y_suffix_vars) | y_suffix_as_int;

        assert!(combined_idx < E_out_s_evals.len(), "combined_idx out of bounds for E_out_s_evals");
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

    /// Performs in-place multilinear extension on ternary evaluation vectors
    /// and updates the temporary accumulator `temp_tA` with the product contribution.
    pub fn compute_and_update_tA_inplace<F: JoltField>(
        ternary_az_evals: &mut [i64], // Size 3^l0, holds Az evals. Initially populated at binary points. Modified in-place.
        ternary_bz_evals: &mut [i64], // Size 3^l0, holds Bz evals. Initially populated at binary points. Modified in-place.
        num_svo_rounds: usize,        // l_0
        e_in_val: F,                  // E_in[x_in] factor for the current x_in
        temp_tA: &mut [F],            // Accumulator vector (size 3^l0), updated with E_in * P_ext contribution.
    ) {
        let num_ternary_points = ternary_az_evals.len();
        let expected_ternary_points = 3_usize.checked_pow(num_svo_rounds as u32)
                                        .expect("Ternary points count overflow");
        debug_assert_eq!(num_ternary_points, expected_ternary_points);
        debug_assert_eq!(ternary_bz_evals.len(), num_ternary_points);
        debug_assert_eq!(temp_tA.len(), num_ternary_points);

        if e_in_val.is_zero() {
            return; // No contribution if E_in is zero
        }

        if num_svo_rounds == 0 {
            if num_ternary_points > 0 { // Should be size 1 if l0=0
                let az0 = ternary_az_evals[0];
                let bz0 = ternary_bz_evals[0];
                if az0 != 0 && bz0 != 0 { // Early check
                    let product_i128 = (az0 as i128) * (bz0 as i128);
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

                let product_p_i128 = (az_final as i128) * (bz_final as i128);
                // product_p_i128 cannot be zero here due to the check above

                // Accumulate contribution from this x_in using mul_i128
                *tA_val += e_in_val.mul_i128(product_p_i128);
            });
    }

    // Distributes the accumulated tA values (sum over x_in) for a single x_out_val 
    // to the appropriate SVO round accumulators.
    pub fn distribute_tA_to_svo_accumulators<F: JoltField>(
        task_tA_accumulator_vec: &[F], 
        x_out_val: usize,             
        num_svo_rounds: usize,
        num_ternary_points: usize,
        E_out_vec: &[Vec<F>],         
        all_idx_mapping_results: &[Vec<(usize, Vec<SVOEvalPoint>, SVOEvalPoint, usize, usize)>], // Updated tuple
        task_svo_accs: &mut Vec<(Vec<F>, Vec<F>)>,
        #[cfg(test)]
        iter_num_x_out_vars: usize, // Added parameter
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
                    #[cfg(test)]
                    iter_num_x_out_vars, 
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
}
