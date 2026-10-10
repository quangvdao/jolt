#![cfg(feature = "test-utils")]

#[path = "../benches/support/word.rs"]
mod word;

#[cfg(test)]
mod tests {
    use super::word::{word_monomial, WordInput};
    use rand_chacha::rand_core::{RngCore, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn monomial_coefficients_equal_subset_parities() {
        let mut rng = ChaCha20Rng::seed_from_u64(84);
        for _ in 0..100 {
            let a = rng.next_u64();
            let b = rng.next_u64();
            for left in 0..8 {
                for right in 0..8 {
                    let mut expected = 0;
                    for window in 0..8 {
                        let coefficient = |word: u64, subset: u32| {
                            (0..8)
                                .filter(|part| part & !subset == 0)
                                .fold(0, |sum, part| sum ^ ((word >> (8 * window + part)) & 1))
                        };
                        expected |= (coefficient(a, left) & coefficient(b, right)) << window;
                    }
                    assert_eq!(word_monomial(WordInput { a, b, left, right }), expected);
                }
            }
        }
    }
}
