#![cfg(feature = "test-utils")]
#![expect(clippy::unwrap_used, reason = "checked test fixtures must construct")]

#[path = "../benches/support/example.rs"]
mod example;

use example::{DenseProductCore, ExampleError};
use jolt_field::{Field, F128};
use jolt_rv64i_kernels::oracle::{mle_at, round_polynomial};
use jolt_sumcheck::ProveRounds;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;

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
