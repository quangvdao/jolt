//! Shared types for G1 group operations.

use crate::field::JoltField;
use crate::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};
use crate::zkvm::witness::{G1AddTerm, TermEnum};
use ark_bn254::Fr;
use ark_ff::{BigInteger, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

/// Public inputs for a single G1 scalar multiplication (the scalar).
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct G1ScalarMulPublicInputs {
    pub scalar: Fr,
}

impl GuestSerialize for G1ScalarMulPublicInputs {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.scalar.guest_serialize(w)
    }
}

impl GuestDeserialize for G1ScalarMulPublicInputs {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            scalar: Fr::guest_deserialize(r)?,
        })
    }
}

impl G1ScalarMulPublicInputs {
    pub fn new(scalar: Fr) -> Self {
        Self { scalar }
    }

    pub fn bits_msb(&self) -> Vec<bool> {
        let scalar_bits_le = self.scalar.into_bigint().to_bits_le();
        (0..256).rev().map(|i| scalar_bits_le[i]).collect()
    }

    pub fn evaluate_bit_mle<F: JoltField>(&self, eval_point: &[F]) -> F {
        let (step_point, pad_sel) = match eval_point.len() {
            8 => (eval_point, F::one()),
            11 => {
                let (step, pad) = eval_point.split_at(8);
                let mut sel = F::one();
                let one = F::one();
                for &p_i in pad {
                    sel *= one - p_i;
                }
                (step, sel)
            }
            _ => panic!(
                "G1ScalarMulPublicInputs::evaluate_bit_mle expected 8 (native) or 11 (padded) vars, got {}",
                eval_point.len()
            ),
        };
        let bits = self.bits_msb();

        let mut evals: Vec<F> = bits
            .iter()
            .map(|&b| if b { F::one() } else { F::zero() })
            .collect();
        debug_assert_eq!(evals.len(), 256);

        let mut len = evals.len();
        for &r_i in step_point {
            let half = len / 2;
            for j in 0..half {
                let a = evals[2 * j];
                let b = evals[2 * j + 1];
                evals[j] = a + r_i * (b - a);
            }
            len = half;
        }
        debug_assert_eq!(len, 1);
        pad_sel * evals[0]
    }
}

/// Values for G1 addition (used during fused addition).
#[derive(Clone, Debug, Default)]
pub struct G1AddValues<F> {
    pub x_p: F,
    pub y_p: F,
    pub ind_p: F,
    pub x_q: F,
    pub y_q: F,
    pub ind_q: F,
    pub x_r: F,
    pub y_r: F,
    pub ind_r: F,
    pub lambda: F,
    pub inv_delta_x: F,
    pub is_double: F,
    pub is_inverse: F,
}

impl<F: JoltField> G1AddValues<F> {
    /// Construct values from a batch of per-term univariate evaluations.
    ///
    /// `poly_evals[t][eval_index]` corresponds to the `t`-th `G1AddTerm` (see `zkvm/witness.rs`)
    /// evaluated at the `eval_index`-th point (0..degree).
    pub fn from_poly_evals<const DEGREE: usize>(
        poly_evals: &[[F; DEGREE]],
        eval_index: usize,
    ) -> Self {
        debug_assert_eq!(
            poly_evals.len(),
            G1AddTerm::COUNT,
            "expected one eval array per G1AddTerm"
        );
        Self {
            x_p: poly_evals[0][eval_index],
            y_p: poly_evals[1][eval_index],
            ind_p: poly_evals[2][eval_index],
            x_q: poly_evals[3][eval_index],
            y_q: poly_evals[4][eval_index],
            ind_q: poly_evals[5][eval_index],
            x_r: poly_evals[6][eval_index],
            y_r: poly_evals[7][eval_index],
            ind_r: poly_evals[8][eval_index],
            lambda: poly_evals[9][eval_index],
            inv_delta_x: poly_evals[10][eval_index],
            is_double: poly_evals[11][eval_index],
            is_inverse: poly_evals[12][eval_index],
        }
    }

    /// Construct values from opened claims, ordered by `G1AddTerm` index.
    pub fn from_claims(claims: &[F]) -> Self {
        debug_assert_eq!(
            claims.len(),
            G1AddTerm::COUNT,
            "expected one claim per G1AddTerm"
        );
        Self {
            x_p: claims[0],
            y_p: claims[1],
            ind_p: claims[2],
            x_q: claims[3],
            y_q: claims[4],
            ind_q: claims[5],
            x_r: claims[6],
            y_r: claims[7],
            ind_r: claims[8],
            lambda: claims[9],
            inv_delta_x: claims[10],
            is_double: claims[11],
            is_inverse: claims[12],
        }
    }

    /// Evaluate the batched G1 add constraint polynomial at this point.
    ///
    /// Uses `delta` to batch the 27 constraint terms: Σ_j δ^j * C_j.
    pub fn eval_constraint(&self, delta: F) -> F {
        let one = F::one();
        let two = F::from_u64(2);
        let three = F::from_u64(3);

        let dx = self.x_q - self.x_p;
        let dy = self.y_q - self.y_p;
        let s_finite = (one - self.ind_p) * (one - self.ind_q);

        // Batch all terms with powers of δ: Σ_j δ^j * term_j
        let mut acc = F::zero();
        let mut delta_pow = F::one();

        // (0) ind_P boolean
        acc += delta_pow * (self.ind_p * (one - self.ind_p));
        delta_pow *= delta;
        // (1) ind_Q boolean
        acc += delta_pow * (self.ind_q * (one - self.ind_q));
        delta_pow *= delta;
        // (2) ind_R boolean
        acc += delta_pow * (self.ind_r * (one - self.ind_r));
        delta_pow *= delta;

        // (3..8) infinity encoding: ind * x = 0, ind * y = 0
        acc += delta_pow * (self.ind_p * self.x_p);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_p * self.y_p);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_q * self.x_q);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_q * self.y_q);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_r * self.x_r);
        delta_pow *= delta;
        acc += delta_pow * (self.ind_r * self.y_r);
        delta_pow *= delta;

        // (9..11) if P = O then R = Q
        acc += delta_pow * (self.ind_p * (self.x_r - self.x_q));
        delta_pow *= delta;
        acc += delta_pow * (self.ind_p * (self.y_r - self.y_q));
        delta_pow *= delta;
        acc += delta_pow * (self.ind_p * (self.ind_r - self.ind_q));
        delta_pow *= delta;

        // (12..14) if Q = O and P != O then R = P
        let q_inf = self.ind_q * (one - self.ind_p);
        acc += delta_pow * (q_inf * (self.x_r - self.x_p));
        delta_pow *= delta;
        acc += delta_pow * (q_inf * (self.y_r - self.y_p));
        delta_pow *= delta;
        acc += delta_pow * (q_inf * (self.ind_r - self.ind_p));
        delta_pow *= delta;

        // (15..16) booleanity of branch bits in finite case
        acc += delta_pow * (s_finite * self.is_double * (one - self.is_double));
        delta_pow *= delta;
        acc += delta_pow * (s_finite * self.is_inverse * (one - self.is_inverse));
        delta_pow *= delta;

        // (17) branch selection: if x_Q = x_P then must be in (double or inverse),
        // else inv_dx must be the inverse of dx (so inv_dx * dx = 1).
        acc += delta_pow
            * (s_finite * (one - self.is_double - self.is_inverse) * (one - self.inv_delta_x * dx));
        delta_pow *= delta;

        // (18..19) if doubling, enforce P == Q
        acc += delta_pow * (s_finite * self.is_double * dx);
        delta_pow *= delta;
        acc += delta_pow * (s_finite * self.is_double * (self.y_q - self.y_p));
        delta_pow *= delta;

        // (20..21) if inverse, enforce P == -Q
        acc += delta_pow * (s_finite * self.is_inverse * dx);
        delta_pow *= delta;
        acc += delta_pow * (s_finite * self.is_inverse * (self.y_q + self.y_p));
        delta_pow *= delta;

        // (22) slope equation (add or double). Inverse case is ungated (vanishes).
        let add_branch = (one - self.is_double - self.is_inverse) * (dx * self.lambda - dy);
        let dbl_branch =
            self.is_double * (two * self.y_p * self.lambda - three * self.x_p * self.x_p);
        acc += delta_pow * (s_finite * (add_branch + dbl_branch));
        delta_pow *= delta;

        // (23) inverse => ind_R = 1
        acc += delta_pow * (s_finite * self.is_inverse * (one - self.ind_r));
        delta_pow *= delta;
        // (24) non-inverse => ind_R = 0
        acc += delta_pow * (s_finite * (one - self.is_inverse) * self.ind_r);
        delta_pow *= delta;

        // (25) x_R formula for non-inverse
        acc += delta_pow
            * (s_finite
                * (one - self.is_inverse)
                * (self.x_r - (self.lambda * self.lambda - self.x_p - self.x_q)));
        delta_pow *= delta;
        // (26) y_R formula for non-inverse
        acc += delta_pow
            * (s_finite
                * (one - self.is_inverse)
                * (self.y_r - (self.lambda * (self.x_p - self.x_r) - self.y_p)));

        acc
    }
}
