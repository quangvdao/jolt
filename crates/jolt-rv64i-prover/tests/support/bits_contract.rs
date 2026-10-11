//! Scheme-independent lifecycle and evaluation checks for the bit-table traits.
use super::RecordedTranscript;
use jolt_field::{One, Zero, F128};
use jolt_rv64i_arith::{BitsRow, BITS_COLUMNS};
use jolt_rv64i_prover::commitment::BitsCommitmentProver;
use jolt_rv64i_verifier::commitment::{BitsGeometry, BitsOpening};
use jolt_rv64i_verifier::points::equality_table;
use jolt_transcript::{Label, Transcript, U64Word};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

pub fn basis(point: &[F128], index: usize) -> F128 {
    point
        .iter()
        .enumerate()
        .map(|(bit, value)| {
            if (index >> bit) & 1 == 0 {
                F128::one() + *value
            } else {
                *value
            }
        })
        .product()
}
pub fn columns(bits: &[BitsRow], cycle: &[F128]) -> Vec<F128> {
    let weights = equality_table(cycle).unwrap();
    (0..BITS_COLUMNS)
        .map(|column| {
            bits.iter()
                .zip(&weights)
                .map(|(row, weight)| {
                    if (row[column / 64] >> (column % 64)) & 1 == 0 {
                        F128::zero()
                    } else {
                        *weight
                    }
                })
                .sum()
        })
        .collect()
}

pub fn lifecycle<S: BitsCommitmentProver<ProverSetup = (), VerifierSetup = ()>>() {
    let mut random = StdRng::seed_from_u64(0x080c_0128);
    for log_T in [3, 6] {
        let geometry = BitsGeometry { log_T };
        let bits: Arc<[BitsRow]> = (0..1_usize << log_T)
            .map(|_| random.gen::<BitsRow>())
            .collect();
        let mut prover = RecordedTranscript::new(b"bits-shared-contract");
        let mut verifier = RecordedTranscript::new(b"bits-shared-contract");
        for transcript in [&mut prover, &mut verifier] {
            transcript.append(&Label(b"params"));
            transcript.append(&U64Word(log_T as u64));
        }
        let (commitment, state) = S::commit(&(), geometry, &bits, &mut prover).unwrap();
        let retained = S::verify_commit(&(), geometry, &commitment, &mut verifier).unwrap();
        assert_eq!(prover.events, verifier.events);
        assert_eq!(prover.state(), verifier.state());
        let cycle = prover.challenge_vector(log_T);
        assert_eq!(cycle, verifier.challenge_vector(log_T));
        let values = columns(&bits, &cycle);
        for value in &values {
            prover.append_labeled(b"opening_claim", value);
            verifier.append_labeled(b"opening_claim", value);
        }
        let rho = prover.challenge_vector(8);
        assert_eq!(rho, verifier.challenge_vector(8));
        let opening = BitsOpening {
            geometry,
            column_point: &rho,
            cycle_point: &cycle,
            columns: &values,
        };
        let column_weights: Vec<_> = (0..BITS_COLUMNS)
            .map(|column| basis(&rho, column))
            .collect();
        let bit_by_bit: F128 = bits
            .iter()
            .enumerate()
            .map(|(j, row)| {
                let cycle_weight = basis(&cycle, j);
                (0..BITS_COLUMNS)
                    .filter(|column| row[column / 64] >> (column % 64) & 1 == 1)
                    .map(|column| column_weights[column] * cycle_weight)
                    .sum::<F128>()
            })
            .sum();
        assert_eq!(opening.value(), bit_by_bit);
        let proof = S::open(&(), state, &opening, &mut prover).unwrap();
        S::verify_opening(&(), retained, &opening, &proof, &mut verifier).unwrap();
        assert_eq!(prover.events, verifier.events);
        assert_eq!(prover.state(), verifier.state());
    }
}
