#![cfg(feature = "test-utils")]

#[path = "../benches/support/merge.rs"]
mod merge;

use jolt_field::F128;
use merge::tree_merge;

#[test]
fn odd_tail_merge_equals_the_xor_of_all_arrays() {
    for workers in [1, 3, 5, 12] {
        let mut arrays: Vec<Vec<F128>> = (0..workers)
            .map(|worker| {
                (0..37)
                    .map(|entry| F128::from_raw((worker * 37 + entry) as u128))
                    .collect()
            })
            .collect();
        let expected: Vec<_> = (0..37)
            .map(|entry| {
                arrays
                    .iter()
                    .fold(F128::from_raw(0), |sum, array| sum + array[entry])
            })
            .collect();
        tree_merge(&mut arrays, 8, Vec::as_mut_slice);
        assert_eq!(arrays[0], expected);
    }
}
