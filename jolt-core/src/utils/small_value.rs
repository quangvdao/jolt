// Subroutines for small-value optimizations in Spartan's first sum-check

pub const NUM_SMALL_VALUE_ROUNDS: usize = 3;

/// Represents a point y in {0, 1, infinity}^ell, where infinity is represented by 2.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TernaryVec(pub Vec<u8>);

impl TernaryVec {
    /// Creates a `TernaryVec` of length `ell` from a base-3 index `idx`.
    pub fn from_index(idx: usize, ell: usize) -> Self {
        let mut y_vec = vec![0u8; ell];
        let mut temp_idx = idx;
        for k in 0..ell {
            let digit = (temp_idx % 3) as u8;
            y_vec[ell - 1 - k] = digit;
            temp_idx /= 3;
        }
        TernaryVec(y_vec)
    }

    /// Computes the base-3 index represented by this `TernaryVec`.
    pub fn local_index(&self) -> usize {
        let mut index = 0;
        let mut power_of_3 = 1;
        for &val in self.0.iter().rev() {
            debug_assert!(val <= 2);
            index += (val as usize) * power_of_3;
            power_of_3 *= 3;
        }
        index
    }

    /// Computes a flattened index for a specific configuration used in Spartan.
    /// Maps `(i, v', u')` to `0..3^ell - 2`, where `v'` is the prefix of `self` of length `i`,
    /// and `u'` is the element at index `i` (must be 0 or 2).
    /// The mapping order is: `i` (0..ell-1), `u'` (0 maps before 2), `v'` (lexicographic base-3).
    pub fn flat_index(&self, i: usize) -> usize {
        debug_assert_eq!(self.0.len(), i + 1);
        let u_prime = self.0[i];
        debug_assert!(u_prime == 0 || u_prime == 2);

        let base_offset: usize = if i == 0 { 0 } else { 3usize.pow(i as u32) - 1 }; // Sum_{k=0}^{i-1} 2*3^k
        let u_offset = if u_prime == 0 {
            0
        } else {
            3usize.pow(i as u32)
        }; // Size of the v' space for u'=0

        // Calculate index of v' within its {0,1,inf}^i space
        let mut v_prime_index = 0;
        let mut power_of_3 = 1;
        for k in 0..i {
            v_prime_index += (self.0[i - 1 - k] as usize) * power_of_3;
            power_of_3 *= 3;
        }

        base_offset + u_offset + v_prime_index
    }

    /// Checks if any element in the vector represents infinity (is 2).
    pub fn has_inf(&self) -> bool {
        self.0.iter().any(|&bit| bit == 2)
    }

    /// Checks if all elements are valid ternary values (0, 1, or 2).
    pub fn _is_valid(&self) -> bool {
        self.0.iter().all(|&ternary_val| ternary_val <= 2)
    }
}

// Added function definition
use crate::poly::sparse_interleaved_poly::SparseCoefficient; // If struct is needed

/// Computes Az(y, x'), Bz(y, x'), Cz(y, x') where y is in {0,1,inf}^k
/// from the interleaved coefficient vector `coeffs`.
/// Placeholder - Requires complex index mapping and difference logic.
pub fn compute_abc_for_ternary_point(
    coeffs: &[SparseCoefficient<i128>],
    y: &TernaryVec,                  // Point in {0,1,inf}^k
    x_prime_idx: usize,              // Index for x' in {0,1}^{num_later_vars}
    num_vars: usize,                 // Total number of variables (k + num_later_vars)
    dense_len_before_binding: usize, // Original dense length (num_coeffs / 3)
) -> (i128, i128, i128) {
    // k = y.0.len()
    // num_later_vars = num_vars - k

    // Base case: y contains no infinity.
    // Map (y, x_prime_idx) to a single linear index z \in {0,1}^num_vars.
    // Read coeffs[3z], coeffs[3z+1], coeffs[3z+2].

    // Recursive case: y contains infinity.
    // Example: y = (y_prefix, inf, y_suffix)
    // Az(y, x') = Az(y_prefix, 1, y_suffix, x') - Az(y_prefix, 0, y_suffix, x')
    // Need recursive calls or an iterative DP approach to compute differences.
    // Need efficient way to find coefficients in the sparse `coeffs` vector.

    // Placeholder implementation
    (0, 0, 0)
}

// Added function definition
/// Computes the idx4 mapping for Algorithm 6 precomputation.
/// For a given prefix y in {0,1,inf}^k, finds all (i, v, u, y_suffix) such that:
/// - i is the first index (from 0 to k-1) where y[i] is binary (0 or 1).
/// - v = y[0..i-1] (prefix before the binary split)
/// - u = y[i] (the first binary value)
/// - y_suffix = y[i+1..k-1] (the rest of the prefix, must ALL be binary)
/// Returns Vec<(round_num, v_prefix, u_current, y_suffix_binary)>
/// Note: round_num is i+1 (1-based). u_current is 0 or 1.
pub fn compute_idx4(y_prefix: &TernaryVec) -> Vec<(usize, TernaryVec, u8, TernaryVec)> {
    let k = y_prefix.0.len();
    let mut results = Vec::new();

    for i in 0..k {
        // i is the potential index of the *first* binary digit (0-based)
        let u = y_prefix.0[i];
        if u == 0 || u == 1 {
            // Found the first binary digit candidate u at index i
            // Check if the suffix y[i+1..k-1] is all binary
            let suffix_is_binary = y_prefix.0[i + 1..].iter().all(|&val| val == 0 || val == 1);

            if suffix_is_binary {
                let v = TernaryVec(y_prefix.0[..i].to_vec()); // Prefix before u
                let y_suffix = TernaryVec(y_prefix.0[i + 1..].to_vec()); // Suffix after u (must be binary)
                let round_num = i + 1; // Rounds are 1-based

                // Convert u=2 (inf) in v to standard ternary for indexing if needed by flat_index
                // let v_standardized = ...;

                results.push((round_num, v, u, y_suffix));

                // Important: idx4 definition implies we only care about the *first*
                // binary split point. If we find one, we don't need to check further i.
                // The pseudocode seems to generate all valid (i,v,u,y) tuples.
                // Let's stick to the pseudocode interpretation: find ALL valid splits.
            }
        } else if u != 2 {
            panic!("Invalid ternary value in y_prefix"); // Should only contain 0, 1, 2
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests the creation of `TernaryVec` from a base-3 index.
    #[test]
    fn test_ternary_vec_from_index() {
        // Test simple cases
        assert_eq!(TernaryVec::from_index(0, 3), TernaryVec(vec![0, 0, 0]));
        assert_eq!(TernaryVec::from_index(1, 3), TernaryVec(vec![0, 0, 1]));
        assert_eq!(TernaryVec::from_index(2, 3), TernaryVec(vec![0, 0, 2]));
        assert_eq!(TernaryVec::from_index(3, 3), TernaryVec(vec![0, 1, 0]));
        assert_eq!(TernaryVec::from_index(4, 3), TernaryVec(vec![0, 1, 1]));

        // Test larger values
        assert_eq!(TernaryVec::from_index(26, 3), TernaryVec(vec![2, 2, 2]));
        assert_eq!(TernaryVec::from_index(13, 3), TernaryVec(vec![1, 1, 1]));
    }

    /// Tests the conversion of `TernaryVec` back to its base-3 index.
    #[test]
    fn test_ternary_vec_local_index() {
        // Test simple cases
        assert_eq!(TernaryVec(vec![0, 0, 0]).local_index(), 0);
        assert_eq!(TernaryVec(vec![0, 0, 1]).local_index(), 1);
        assert_eq!(TernaryVec(vec![0, 0, 2]).local_index(), 2);
        assert_eq!(TernaryVec(vec![0, 1, 0]).local_index(), 3);

        // Test larger values
        assert_eq!(TernaryVec(vec![1, 1, 1]).local_index(), 13);
        assert_eq!(TernaryVec(vec![2, 2, 2]).local_index(), 26);

        // Verify round-trip
        for idx in 0..27 {
            let tv = TernaryVec::from_index(idx, 3);
            assert_eq!(tv.local_index(), idx);
        }
    }

    /// Tests the flattened index calculation used in Spartan.
    #[test]
    fn test_ternary_vec_flat_index() {
        // Test for i=0
        // self=[0], u\'=0, v\'=[]. base=3^0-1=0, u_off=0, v\'_idx=0. Total = 0.
        assert_eq!(TernaryVec(vec![0]).flat_index(0), 0);
        // self=[2], u\'=2, v\'=[]. base=0, u_off=3^0=1, v\'_idx=0. Total = 1.
        assert_eq!(TernaryVec(vec![2]).flat_index(0), 1);

        // Test for i=1
        // base_offset = 3^1 - 1 = 2
        // self=[0, 0], u\'=0, v\'=[0]. u_off=0, v\'_idx=0. Total = 2 + 0 + 0 = 2.
        assert_eq!(TernaryVec(vec![0, 0]).flat_index(1), 2);
        // self=[1, 0], u\'=0, v\'=[1]. u_off=0, v\'_idx=1. Total = 2 + 0 + 1 = 3.
        assert_eq!(TernaryVec(vec![1, 0]).flat_index(1), 3);
        // self=[2, 0], u\'=0, v\'=[2]. u_off=0, v\'_idx=2. Total = 2 + 0 + 2 = 4.
        assert_eq!(TernaryVec(vec![2, 0]).flat_index(1), 4);
        // self=[0, 2], u\'=2, v\'=[0]. u_off=3^1=3, v\'_idx=0. Total = 2 + 3 + 0 = 5.
        assert_eq!(TernaryVec(vec![0, 2]).flat_index(1), 5);
        // self=[1, 2], u\'=2, v\'=[1]. u_off=3^1=3, v\'_idx=1. Total = 2 + 3 + 1 = 6.
        assert_eq!(TernaryVec(vec![1, 2]).flat_index(1), 6);
        // self=[2, 2], u\'=2, v\'=[2]. u_off=3^1=3, v\'_idx=2. Total = 2 + 3 + 2 = 7.
        assert_eq!(TernaryVec(vec![2, 2]).flat_index(1), 7);

        // Test for i=2
        // base_offset = 3^2 - 1 = 8
        // self=[0, 0, 0], u\'=0, v\'=[0, 0]. u_off=0, v\'_idx=0. Total = 8 + 0 + 0 = 8.
        assert_eq!(TernaryVec(vec![0, 0, 0]).flat_index(2), 8);
        // self=[1, 0, 0], u\'=0, v\'=[1, 0]. u_off=0, v\'_idx=(0*1 + 1*3)=3. Total = 8 + 0 + 3 = 11.
        assert_eq!(TernaryVec(vec![1, 0, 0]).flat_index(2), 11);
        // self=[0, 1, 0], u\'=0, v\'=[0, 1]. u_off=0, v\'_idx=(1*1 + 0*3)=1. Total = 8 + 0 + 1 = 9.
        assert_eq!(TernaryVec(vec![0, 1, 0]).flat_index(2), 9);
        // self=[0, 0, 2], u\'=2, v\'=[0, 0]. u_off=3^2=9, v\'_idx=0. Total = 8 + 9 + 0 = 17.
        assert_eq!(TernaryVec(vec![0, 0, 2]).flat_index(2), 17);
    }

    /// Tests the check for the presence of infinity (2) in the vector.
    #[test]
    fn test_ternary_vec_has_inf() {
        assert!(!TernaryVec(vec![0, 0, 0]).has_inf());
        assert!(!TernaryVec(vec![0, 1, 1]).has_inf());
        assert!(TernaryVec(vec![0, 2, 0]).has_inf());
        assert!(TernaryVec(vec![2, 0, 0]).has_inf());
        assert!(TernaryVec(vec![0, 0, 2]).has_inf());
    }
}
