use super::transcript::Transcript;
use crate::field::JoltField;
use ark_ec::{AffineRepr, CurveGroup};
use ark_serialize::CanonicalSerialize;
use blake2::digest::consts::U32;
use blake2::{Blake2b, Digest};

type Blake2b256 = Blake2b<U32>;

/// Represents the current state of the protocol's Fiat-Shamir transcript using Blake2b.
#[derive(Default, Clone)]
pub struct Blake2bTranscript {
    /// 256-bit running state
    pub state: [u8; 32],
    /// We append an ordinal to each invocation of the hash
    n_rounds: u32,
    #[cfg(test)]
    /// A complete history of the transcript's `state`; used for testing.
    state_history: Vec<[u8; 32]>,
    #[cfg(test)]
    /// For a proof to be valid, the verifier's `state_history` should always match
    /// the prover's. In testing, the Jolt verifier may be provided the prover's
    /// `state_history` so that we can detect any deviations and the backtrace can
    /// tell us where it happened.
    expected_state_history: Option<Vec<[u8; 32]>>,
    #[cfg(test)]
    last_label: &'static [u8],
}

impl Blake2bTranscript {
    fn hasher(&self) -> Blake2b256 {
        let mut packed = [0_u8; 32];
        packed[28..].copy_from_slice(&self.n_rounds.to_be_bytes());
        Blake2b256::new()
            .chain_update(self.state)
            .chain_update(packed)
    }

    fn challenge_bytes(&mut self, out: &mut [u8]) {
        let mut remaining_len = out.len();
        let mut start = 0;
        while remaining_len > 32 {
            self.challenge_bytes32(&mut out[start..start + 32]);
            start += 32;
            remaining_len -= 32;
        }
        let mut full_rand = [0_u8; 32];
        self.challenge_bytes32(&mut full_rand);
        out[start..start + remaining_len].copy_from_slice(&full_rand[..remaining_len]);
    }

    // Loads exactly 32 bytes from the transcript by hashing the seed with the round constant
    fn challenge_bytes32(&mut self, out: &mut [u8]) {
        assert_eq!(32, out.len());
        let rand: [u8; 32] = self.hasher().finalize().into();
        out.clone_from_slice(rand.as_slice());
        self.update_state(rand);
    }

    fn update_state(&mut self, new_state: [u8; 32]) {
        self.state = new_state;
        self.n_rounds += 1;
        #[cfg(test)]
        {
            if let Some(expected_state_history) = &self.expected_state_history {
                if new_state != expected_state_history[self.n_rounds as usize] {
                    eprintln!(
                        "[FS MISMATCH] round={}, last_label={:?}",
                        self.n_rounds,
                        std::str::from_utf8(self.last_label).unwrap_or("???"),
                    );
                    panic!("Fiat-Shamir transcript mismatch at round {}", self.n_rounds);
                }
            }
            self.state_history.push(new_state);
        }
    }
}

impl Transcript for Blake2bTranscript {
    fn new(label: &'static [u8]) -> Self {
        assert!(label.len() < 33);
        let mut padded = [0_u8; 32];
        padded[..label.len()].copy_from_slice(label);
        let out = Blake2b256::new().chain_update(padded).finalize();

        Self {
            state: out.into(),
            n_rounds: 0,
            #[cfg(test)]
            state_history: vec![out.into()],
            #[cfg(test)]
            expected_state_history: None,
            #[cfg(test)]
            last_label: b"",
        }
    }

    #[cfg(test)]
    /// Compare this transcript to `other` and panic if/when they deviate.
    /// Typically used to compare the verifier's transcript to the prover's.
    fn compare_to(&mut self, other: Self) {
        self.expected_state_history = Some(other.state_history);
    }

    fn raw_append_label(&mut self, label: &'static [u8]) {
        assert!(label.len() < 33);
        #[cfg(test)]
        {
            self.last_label = label;
        }
        let mut padded = [0_u8; 32];
        padded[..label.len()].copy_from_slice(label);
        let hasher = self.hasher().chain_update(padded);
        self.update_state(hasher.finalize().into());
    }

    fn raw_append_bytes(&mut self, bytes: &[u8]) {
        // Add the message and label
        let hasher = self.hasher().chain_update(bytes);
        self.update_state(hasher.finalize().into());
    }

    fn raw_append_u64(&mut self, x: u64) {
        let mut packed = [0_u8; 32];
        packed[24..].copy_from_slice(&x.to_be_bytes());
        let hasher = self.hasher().chain_update(packed);
        self.update_state(hasher.finalize().into());
    }

    fn raw_append_scalar<F: JoltField>(&mut self, scalar: &F) {
        let mut buf = [0u8; 32];
        let size = scalar.serialized_size(ark_serialize::Compress::No);
        scalar.serialize_uncompressed(&mut buf[..size]).unwrap();
        buf[..size].reverse();
        self.raw_append_bytes(&buf[..size]);
    }

    fn raw_append_point<G: CurveGroup>(&mut self, point: &G) {
        if point.is_zero() {
            self.raw_append_bytes(&[0_u8; 64]);
            return;
        }

        let aff = point.into_affine();
        let mut x_bytes = [0u8; 32];
        let mut y_bytes = [0u8; 32];
        let x = aff.x().unwrap();
        let x_size = x.serialized_size(ark_serialize::Compress::Yes);
        x.serialize_compressed(&mut x_bytes[..x_size]).unwrap();
        x_bytes[..x_size].reverse();
        let y = aff.y().unwrap();
        let y_size = y.serialized_size(ark_serialize::Compress::Yes);
        y.serialize_compressed(&mut y_bytes[..y_size]).unwrap();
        y_bytes[..y_size].reverse();

        let hasher = self
            .hasher()
            .chain_update(&x_bytes[..x_size])
            .chain_update(&y_bytes[..y_size]);
        self.update_state(hasher.finalize().into());
    }

    fn challenge_u128(&mut self) -> u128 {
        let mut buf = [0u8; 16];
        self.challenge_bytes(&mut buf);
        buf.reverse();
        u128::from_be_bytes(buf)
    }

    fn challenge_scalar<F: JoltField>(&mut self) -> F {
        // Under the hood all Fr are 128 bits for performance
        self.challenge_scalar_128_bits()
    }

    fn challenge_scalar_128_bits<F: JoltField>(&mut self) -> F {
        let mut buf = [0u8; 16];
        self.challenge_bytes(&mut buf);
        buf.reverse();
        F::from_bytes(&buf)
    }

    fn challenge_vector<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        (0..len)
            .map(|_i| self.challenge_scalar())
            .collect::<Vec<F>>()
    }

    // Compute powers of scalar q : (1, q, q^2, ..., q^(len-1))
    fn challenge_scalar_powers<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        let q: F = self.challenge_scalar();
        let mut q_powers = vec![F::one(); len];
        for i in 1..len {
            q_powers[i] = q_powers[i - 1] * q;
        }
        q_powers
    }

    fn challenge_scalar_optimized<F: JoltField>(&mut self) -> F::Challenge {
        // The smaller challenge which is then converted into a
        // MontU128Challenge
        let challenge_scalar: u128 = self.challenge_u128();
        F::Challenge::from(challenge_scalar)
    }

    fn challenge_vector_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F::Challenge> {
        (0..len)
            .map(|_i| self.challenge_scalar_optimized::<F>())
            .collect::<Vec<F::Challenge>>()
    }

    fn challenge_scalar_powers_optimized<F: JoltField>(&mut self, len: usize) -> Vec<F> {
        // This is still different from challenge_scalar_powers as inside the for loop
        // we use an optimised multiplication every time we compute the powers.
        let q: F::Challenge = self.challenge_scalar_optimized::<F>();
        let mut q_powers = vec![<F as ark_std::One>::one(); len];
        for i in 1..len {
            q_powers[i] = q * q_powers[i - 1]; // this is optimised
        }
        q_powers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use std::collections::HashSet;

    #[test]
    fn test_challenge_scalar_128_bits() {
        let mut transcript = Blake2bTranscript::new(b"test_128_bit_scalar");
        let mut scalars = HashSet::new();

        for i in 0..10000 {
            let scalar: Fr = transcript.challenge_scalar_128_bits();

            let num_bits = scalar.num_bits();
            assert!(
                num_bits <= 128,
                "Scalar at iteration {i} has {num_bits} bits, expected <= 128",
            );

            assert!(
                scalars.insert(scalar),
                "Duplicate scalar found at iteration {i}",
            );
        }
    }

    #[test]
    fn test_challenge_special_trivial() {
        use ark_std::UniformRand;
        let mut rng = ark_std::test_rng();
        let mut transcript1 = Blake2bTranscript::new(b"test_trivial_challenge");

        let challenge = transcript1.challenge_scalar_optimized::<Fr>();
        // The same challenge as a full fat Fr element
        let challenge_regular: Fr = challenge.into();

        let field_elements: Vec<Fr> = (0..10).map(|_| Fr::rand(&mut rng)).collect();

        for (i, &field_elem) in field_elements.iter().enumerate() {
            let result_challenge = field_elem * challenge;
            let result_regular = field_elem * challenge_regular;

            assert_eq!(
                result_challenge, result_regular,
                "Multiplication mismatch at index {i}"
            );
        }

        let field_elem = Fr::rand(&mut rng);
        #[allow(clippy::op_ref)]
        let result_ref = field_elem * &challenge;
        let result_regular = field_elem * challenge;
        assert_eq!(
            result_ref, result_regular,
            "Reference multiplication mismatch"
        );
    }
}
