#![cfg(feature = "test-utils")]
#![expect(
    clippy::panic,
    clippy::unwrap_used,
    reason = "test fixtures and assertions may panic on failure"
)]

#[path = "../benches/support/example.rs"]
mod example;

use example::{DenseProductCore, ExampleError};
use jolt_field::{Field, F128};
use jolt_poly::CompressedPoly;
use jolt_rv64i_kernels::oracle::{mle_at, round_polynomial};
use jolt_sumcheck::{
    prove_batch, BatchMember, BatchPrelude, BooleanHypercube, ClearProof, ClearSumcheckRecorder,
    EvaluationClaim, ProveRounds, ProvedBatch, SequentialRounds, SumcheckClaim, SumcheckError,
    SumcheckProof, SumcheckRecorder, SumcheckVerifier, OPENING_CLAIM_TRANSCRIPT_LABEL,
    SUMCHECK_CLAIM_TRANSCRIPT_LABEL, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

const LABEL: &[u8] = b"packed-dense-product";

struct BatchFixture {
    a: Vec<F128>,
    b: Vec<F128>,
    prelude: BatchPrelude<F128>,
    proved: ProvedBatch<F128>,
    proof: SumcheckProof<F128, ()>,
    final_values: [F128; 2],
    transcript_state: [u8; 32],
}

#[derive(Debug)]
enum VerificationFailure {
    Sumcheck(SumcheckError<F128>),
    FinalOracleEvaluationMismatch,
}

impl BatchFixture {
    fn prove(log_t: usize) -> Self {
        let mut rng = ChaCha20Rng::seed_from_u64(0x0070_726f_6475_6374 + log_t as u64);
        let a: Vec<_> = (0..1 << log_t).map(|_| F128::random(&mut rng)).collect();
        let b: Vec<_> = (0..1 << log_t).map(|_| F128::random(&mut rng)).collect();
        let mut core = DenseProductCore::new(a.clone(), b.clone()).unwrap();
        let input_claim: F128 = a.iter().zip(&b).map(|(&a, &b)| a * b).sum();
        assert_eq!(core.initial_claim(), input_claim);
        let mut recorder = ClearSumcheckRecorder::<F128>::new();
        let mut transcript = Blake2bTranscript::<F128>::new(LABEL);
        recorder.absorb_input_claims(&[input_claim], &mut transcript);
        let prelude = BatchPrelude::try_new(
            vec![BatchMember {
                input_claim,
                coefficient: transcript.challenge_scalar(),
                rounds: log_t,
                offset: 0,
            }],
            log_t,
            2,
        )
        .unwrap();
        let proved = prove_batch(
            &prelude,
            &mut [&mut core],
            &mut SequentialRounds,
            &mut recorder,
            &mut transcript,
        )
        .unwrap();
        let recorded = recorder
            .finish(&proved.member_claims, &mut transcript)
            .unwrap();
        Self {
            a,
            b,
            prelude,
            proved,
            proof: recorded.proof,
            final_values: core.final_values().unwrap(),
            transcript_state: transcript.state(),
        }
    }

    fn verify(
        &self,
        proof: &SumcheckProof<F128, ()>,
    ) -> Result<EvaluationClaim<F128>, VerificationFailure> {
        let mut transcript = Blake2bTranscript::<F128>::new(LABEL);
        let member = &self.prelude.members[0];
        transcript.append_labeled(SUMCHECK_CLAIM_TRANSCRIPT_LABEL, &member.input_claim);
        assert_eq!(transcript.challenge_scalar(), member.coefficient);
        let SumcheckProof::Clear(ClearProof::Compressed(proof)) = proof else {
            unreachable!("the clear recorder produces compressed rounds")
        };
        let reduced = SumcheckVerifier::verify_compressed(
            &SumcheckClaim::new(self.prelude.max_num_vars, 2, self.prelude.claimed_sum),
            proof,
            BooleanHypercube,
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            &mut transcript,
        )
        .map_err(VerificationFailure::Sumcheck)?;
        let point = reduced.point.as_slice();
        let native_value = mle_at(&self.a, point).unwrap() * mle_at(&self.b, point).unwrap();
        if reduced.value != member.coefficient * native_value {
            return Err(VerificationFailure::FinalOracleEvaluationMismatch);
        }
        transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, &native_value);
        assert_eq!(transcript.state(), self.transcript_state);
        Ok(reduced)
    }
}

#[test]
fn dense_product_batch_messages_match_oracle_and_proof_verifies() {
    for log_t in 1..=8 {
        let fixture = BatchFixture::prove(log_t);
        let coefficient = fixture.prelude.members[0].coefficient;
        let SumcheckProof::Clear(ClearProof::Compressed(proof)) = &fixture.proof else {
            unreachable!("the clear recorder produces compressed rounds")
        };
        assert_eq!(proof.round_polynomials.len(), log_t);
        let mut claim = fixture.prelude.claimed_sum;
        for (round, message) in proof.round_polynomials.iter().enumerate() {
            let expected = round_polynomial(
                &[&fixture.a, &fixture.b],
                &fixture.proved.challenges[..round],
                2,
                |leaves| coefficient * leaves[0] * leaves[1],
            )
            .unwrap();
            assert_eq!(message.decompress(claim), expected);
            claim = expected.evaluate(fixture.proved.challenges[round]);
        }
        let expected = [
            mle_at(&fixture.a, &fixture.proved.challenges).unwrap(),
            mle_at(&fixture.b, &fixture.proved.challenges).unwrap(),
        ];
        assert_eq!(fixture.final_values, expected);
        assert_eq!(fixture.proved.member_claims, [expected[0] * expected[1]]);
        assert_eq!(
            fixture.proved.final_claim,
            coefficient * expected[0] * expected[1]
        );
        assert_eq!(claim, fixture.proved.final_claim);
        let reduced = fixture.verify(&fixture.proof).unwrap();
        assert_eq!(reduced.point.as_slice(), fixture.proved.challenges);
        assert_eq!(reduced.value, fixture.proved.final_claim);
    }
}

#[test]
fn changed_compressed_round_coefficient_fails_final_oracle_evaluation_check() {
    let fixture = BatchFixture::prove(8);
    let mut altered = fixture.proof.clone();
    let SumcheckProof::Clear(ClearProof::Compressed(proof)) = &mut altered else {
        unreachable!("the clear recorder produces compressed rounds")
    };
    let round = proof.round_polynomials.last_mut().unwrap();
    let mut coefficients = round.coeffs_except_linear_term().to_vec();
    coefficients[0] += F128::from_raw(1);
    *round = CompressedPoly::new(coefficients);
    match fixture.verify(&altered) {
        Err(VerificationFailure::FinalOracleEvaluationMismatch) => {}
        Err(VerificationFailure::Sumcheck(error)) => {
            panic!("the compressed reduction must reach the final oracle check: {error:?}")
        }
        Ok(_) => panic!("the final oracle check accepted a changed coefficient"),
    }
}

#[test]
fn dense_product_messages_and_final_values_equal_the_defining_sum() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x0070_726f_6475_6374);
    for log_t in 1..=8 {
        let a: Vec<_> = (0..1 << log_t).map(|_| F128::random(&mut rng)).collect();
        let b: Vec<_> = (0..1 << log_t).map(|_| F128::random(&mut rng)).collect();
        let mut core = DenseProductCore::new(a.clone(), b.clone()).unwrap();
        let mut claim: F128 = a.iter().zip(&b).map(|(&a, &b)| a * b).sum();
        assert_eq!(core.initial_claim(), claim);
        let mut point = Vec::new();
        for round in 0..log_t {
            let expected =
                round_polynomial(&[&a, &b], &point, 2, |leaves| leaves[0] * leaves[1]).unwrap();
            let message = core
                .prove_round(point.last().copied(), round, claim)
                .unwrap();
            assert_eq!(message, expected);
            let challenge = F128::random(&mut rng);
            claim = message.evaluate(challenge);
            point.push(challenge);
        }
        core.finish_rounds(*point.last().unwrap()).unwrap();
        let expected = [mle_at(&a, &point).unwrap(), mle_at(&b, &point).unwrap()];
        assert_eq!(core.final_values().unwrap(), expected);
        assert_eq!(claim, expected[0] * expected[1]);
    }
}

#[test]
fn malformed_dense_product_calls_return_errors_without_rebinding() {
    let one = F128::from_raw(1);
    assert!(matches!(
        DenseProductCore::new(vec![], vec![]),
        Err(ExampleError::InvalidLength { length: 0 })
    ));
    assert!(matches!(
        DenseProductCore::new(vec![one; 3], vec![one; 3]),
        Err(ExampleError::InvalidLength { length: 3 })
    ));
    assert!(matches!(
        DenseProductCore::new(vec![one; 2], vec![one; 4]),
        Err(ExampleError::LengthMismatch { .. })
    ));
    let mut core = DenseProductCore::new(vec![one; 4], vec![one; 4]).unwrap();
    assert!(core.final_values().is_err());
    assert!(core.prove_round(None, 1, F128::from_raw(0)).is_err());
    let round = core.prove_round(None, 0, core.initial_claim()).unwrap();
    let challenge = F128::from_raw(0x71);
    let claim = round.evaluate(challenge);
    assert!(core.prove_round(Some(challenge), 1, claim + one).is_err());
    assert!(core.prove_round(Some(challenge), 1, claim).is_err());
    assert!(core.finish_rounds(challenge).is_err());
}
